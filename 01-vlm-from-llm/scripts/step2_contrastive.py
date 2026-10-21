"""실행 2 — SigLIP zero-shot 분류·이미지-텍스트 검색, 그리고 softmax vs sigmoid 손실 비교.

    uv run python scripts/step2_contrastive.py                 # 전부
    uv run python scripts/step2_contrastive.py --part zeroshot  # zeroshot | padding | retrieval | train 중 하나

보조 코드(데이터 내려받기, SigLIP 불러오기, 출력 벡터 계산과 캐시, 미니배치 순회)는 준비돼 있다.
`TODO(직접)` 표시가 있는 함수와 main()의 각 부분을 채운다. 예측·관측·결과는 README "실행 기록"에 쓴다.

데이터(처음 한 번 내려받음): STL-10 시험 세트 142MB, Flickr8k 시험 137MB, Flickr8k 학습 830MB(train 부분에서만).
출력 벡터는 data/features/에 저장해 두므로 두 번째 실행부터는 인코더를 다시 돌리지 않는다.
"""

import argparse
from pathlib import Path

import torch
import transformers
from common import pick_device, print_table
from datasets import load_dataset
from transformers import AutoProcessor, SiglipModel
from random import random, randint
import math
import sys
import copy

transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()

HERE = Path(__file__).resolve().parents[1]
FEATURES = HERE / "data" / "features"  # 저장소 .gitignore 대상

MODEL = "google/siglip-so400m-patch14-224"  # 실행 1과 같은 체크포인트. 텍스트 최대 16토큰
STL10 = "tanganke/stl10"  # 10개 클래스, 96×96 사진
FLICKR8K = "jxie/flickr8k"  # 이미지 한 장에 캡션 5개


# ---------------------------------------------------------------- 보조 코드


def load_siglip(device: torch.device):
    """이미지·텍스트 인코더를 모두 가진 SigLIP을 float32로 불러온다. 반환: (model, processor)"""
    model = SiglipModel.from_pretrained(MODEL).to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL)
    return model, processor


def scale_and_bias(model) -> tuple[float, float]:
    """사전학습된 온도 t와 편향 b. 모델은 log t를 저장하므로 exp를 취한다. 점수 = t·⟨이미지, 텍스트⟩ + b"""
    return model.logit_scale.exp().item(), model.logit_bias.item()


def load_stl10(n: int | None = 1000, seed: int = 0):
    """STL-10 시험 세트에서 n장을 고정된 순서로 뽑는다. 반환: (PIL 이미지 목록, 라벨 텐서, 클래스 이름 목록)"""
    ds = load_dataset(STL10, data_files={"test": "data/test-*.parquet"}, split="test")
    classes = ["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"]
    if n is not None:
        ds = ds.shuffle(seed=seed).select(range(n))
    return [im.convert("RGB") for im in ds["image"]], torch.tensor(ds["label"]), classes


def load_flickr8k(split: str = "test", n: int | None = None):
    """Flickr8k의 한 분할. 반환: (PIL 이미지 N장, 캡션 5N개, 캡션별 이미지 번호 텐서 (5N,))

    캡션은 이미지 0의 캡션 5개, 이미지 1의 캡션 5개, ... 순서로 펼쳐져 있다.
    """
    files = {"train": "data/train-*.parquet", "validation": "data/validation-*.parquet", "test": "data/test-*.parquet"}
    ds = load_dataset(FLICKR8K, data_files={split: files[split]}, split=split)
    if n is not None:
        ds = ds.select(range(n))
    images = [im.convert("RGB") for im in ds["image"]]
    captions = [ds[f"caption_{k}"][i] for i in range(len(ds)) for k in range(5)]
    caption_to_image = torch.arange(len(ds)).repeat_interleave(5)
    return images, captions, caption_to_image


@torch.inference_mode()
def encode_images(model, processor, images, device: torch.device, batch_size: int = 32) -> torch.Tensor:
    """이미지 인코더의 출력 벡터(정규화됨, CPU float32). 모양 (N, D)"""
    out = []
    for i in range(0, len(images), batch_size):
        pixel_values = processor(images=images[i : i + batch_size], return_tensors="pt")["pixel_values"].to(device)
        out.append(model.get_image_features(pixel_values=pixel_values).pooler_output.float().cpu())
    return torch.nn.functional.normalize(torch.cat(out), dim=-1)


@torch.inference_mode()
def encode_texts(
    model, processor, texts, device: torch.device, padding="max_length", batch_size: int = 64
) -> torch.Tensor:
    """텍스트 인코더의 출력 벡터(정규화됨, CPU float32). 모양 (M, D)

    padding: "max_length"(학습 때와 같은 16칸) | "longest"(배치 안 가장 긴 문장에 맞춤) | False(패딩 없음)
    16토큰을 넘는 캡션은 잘린다(truncation). 패딩 없음은 길이가 제각각이라 한 문장씩 넣는다.
    """
    step = 1 if padding is False else batch_size
    out = []
    for i in range(0, len(texts), step):
        enc = processor(text=texts[i : i + step], padding=padding, truncation=True, return_tensors="pt")
        out.append(model.get_text_features(input_ids=enc["input_ids"].to(device)).pooler_output.float().cpu())
    return torch.nn.functional.normalize(torch.cat(out), dim=-1)


def cached(name: str, fn) -> torch.Tensor:
    """data/features/<name>.pt가 있으면 불러오고, 없으면 fn()을 실행해 저장한다."""
    path = FEATURES / f"{name}.pt"
    if path.exists():
        return torch.load(path)
    FEATURES.mkdir(parents=True, exist_ok=True)
    out = fn()
    torch.save(out, path)
    return out


def pair_batches(n_images: int, batch_size: int, generator: torch.Generator, captions_per_image: int = 5):
    """한 에폭 동안 (이미지 번호, 캡션 번호) 미니배치를 돌려준다. 이미지마다 캡션 하나를 무작위로 고른다.

    한 배치에 같은 이미지가 두 번 들어가면, 그 이미지의 다른 캡션이 오답으로 취급되는 문제(거짓 오답)가 생긴다.
    그래서 한 배치 안에서는 이미지가 겹치지 않게 한다. 남는 꼬리 배치는 버린다(배치 크기를 정확히 지키려고).
    """
    order = torch.randperm(n_images, generator=generator)
    pick = torch.randint(captions_per_image, (n_images,), generator=generator)
    for i in range(0, n_images - batch_size + 1, batch_size):
        img = order[i : i + batch_size]
        yield img, img * captions_per_image + pick[img]


# ---------------------------------------------------------------- 직접 구현


def zero_shot(img: torch.Tensor, txt: torch.Tensor, t: float, b: float) -> dict:
    """zero-shot 분류. img (N, D), txt (C, D)는 정규화된 출력 벡터, C는 클래스 수.

    TODO(직접): 점수를 클래스별 확률로 바꾸고(예측 Q1), 예측 클래스를 정한다.
    반환: {"probs": (N, C) 텐서, "pred": (N,) 텐서}
    """
    print("img", img.size(), "txt", txt.size())
    logits = (img @ txt.T) * t + b # (N,C)
    probs = torch.sigmoid(logits)
    print("sum of probs", torch.sum(probs, dim=1).mean())
    print(probs)
    pred = torch.argmax(probs, dim=1)
    return {"probs": probs, "pred": pred}


def recall_at_k(img: torch.Tensor, txt: torch.Tensor, caption_to_image: torch.Tensor, ks=(1, 5, 10)) -> dict:
    """이미지-텍스트 검색의 recall@k. img (N, D), txt (5N, D), caption_to_image (5N,)

    TODO(직접): 이미지→텍스트(정답 캡션 5개 중 하나라도 상위 k개에 들면 성공)와
    텍스트→이미지(정답 이미지가 상위 k개에 들면 성공)를 각각 계산한다.
    반환: {"i2t@1": float, "i2t@5": ..., "t2i@1": ..., ..., "n_images": N, "n_captions": 5N}
    """
    N = img.size()[0]
    image_to_caption = torch.arange(5 * N).view(N, 1, 5)
    logits = (img @ txt.T) # (N, 5N)
    return_dict = {"n_images": N, "n_captions": 5*N}
    for k in ks:
        #i2t
        _, i2t = torch.topk(logits, k=k, dim=1) # (N,k)
        _, t2i = torch.topk(logits, k=k, dim=0) # (k, 5N)
        i2t_recall = (i2t.unsqueeze(2) == image_to_caption).any(dim=[-2,-1]).sum() / N
        t2i_recall = (t2i == caption_to_image.unsqueeze(0)).any(dim=0).sum() / (5*N)
        return_dict[f"i2t@{k}"] = i2t_recall.item()
        return_dict[f"t2i@{k}"] = t2i_recall.item()
    return return_dict


class ProjectionHead(torch.nn.Module):
    """얼린 출력 벡터 위에 학습하는 작은 사영. 이미지와 텍스트 각각에 하나씩 둔다.

    TODO(직접): 구조(선형 한 층 등), 출력 차원, 출력 정규화 여부를 정한다.
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self._n_hidden = 2048
        self._project_layer = torch.nn.Sequential(
                torch.nn.Linear(dim_in, self._n_hidden),
                torch.nn.BatchNorm1d(self._n_hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(self._n_hidden, dim_out)
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._project_layer(x)


def softmax_loss(img: torch.Tensor, txt: torch.Tensor, log_t: torch.Tensor) -> torch.Tensor:
    """CLIP식 대조 손실. img·txt (B, d)는 정규화된 벡터이고 같은 행 번호끼리 정답 쌍.

    TODO(직접): 양방향 교차 엔트로피.
    """
    B, d = img.size()
    t = torch.exp(log_t)
    logits = img @ txt.T * t
    labels = torch.arange(B, device=img.device)
    crit = torch.nn.CrossEntropyLoss()
    loss_i = crit(logits, labels)
    loss_t = crit(logits.T, labels)
    return (loss_i + loss_t) / 2

def sigmoid_loss(img: torch.Tensor, txt: torch.Tensor, log_t: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """SigLIP식 쌍별 시그모이드 손실. img·txt (B, d)는 정규화된 벡터이고 같은 행 번호끼리 정답 쌍.

    TODO(직접): 모든 쌍에 대한 이진 분류 손실. 배치 크기로 어떻게 나눌지도 정한다.
    """
    B, d = img.size()
    t = torch.exp(log_t)
    logits = img @ txt.T * t + b
    labels = torch.eye(B, device=img.device) * 2 - torch.ones(B, device=img.device)
    return -torch.sum(torch.nn.functional.logsigmoid(labels*logits)) / B

def train_heads(loss_name: str, img: torch.Tensor, txt: torch.Tensor, va_img: torch.Tensor, va_txt: torch.Tensor, va_cap2img: torch.Tensor, batch_size: int, steps: int, seed: int = 0):
    """얼린 출력 벡터 위에서 ProjectionHead 두 개(이미지·텍스트)를 학습한다.

    img (N, D)는 학습 이미지, txt (5N, D)는 학습 캡션. 미니배치는 pair_batches()로 만든다.
    TODO(직접): 온도·편향을 학습 가능하게 둘지와 초기값, 옵티마이저와 학습률을 정한다.
    두 손실의 비교는 같은 데이터·같은 head·같은 스텝 수에서 배치 크기만 바꾼다(GUIDE 검증 기준).
    반환: (이미지 head, 텍스트 head)
    """
    device = img.device

    log_t = torch.nn.Parameter(torch.tensor(math.log(10),device=device)) #static value for comparison
    b = torch.nn.Parameter(torch.tensor(-10.0,device=device))

    if loss_name == "softmax":
        crit = softmax_loss 
    elif loss_name == "sigmoid":
        crit = lambda i,t,lt: sigmoid_loss(i,t,lt,b)
    else:
        raise NotImplementedError

    D = img.size()[-1]
    dim_out = 256
  
    def endless(n_images, batch_size, generator):
        while True:
            yield from pair_batches(n_images, batch_size, generator)

    it = endless(len(img), batch_size, torch.Generator().manual_seed(seed))
    img_head = ProjectionHead(D, dim_out).to(device)
    txt_head = ProjectionHead(D, dim_out).to(device)

    @torch.inference_mode()
    def validate():
        img_head.eval()
        txt_head.eval()
        with torch.inference_mode():
            img = torch.nn.functional.normalize(img_head(va_img),dim=-1)
            txt = torch.nn.functional.normalize(txt_head(va_txt),dim=-1)
        img_head.train()
        txt_head.train()
        recall = recall_at_k(img, txt, va_cap2img)
        return (recall.get("i2t@1", 0.0) + recall.get("t2i@1", 0.0))/2

    #optimizer
    optimizer = torch.optim.AdamW(
        [
            {"params": img_head.parameters(), "weight_decay": 0.01},
            {"params": txt_head.parameters(), "weight_decay": 0.01},
            {"params": [log_t, b], "weight_decay": 0.0 }
        ],
        lr=1e-3,
    )

    #scheduler
    max_steps = 50_000
    eval_loss = False
    if steps < 0:
        eval_loss = True
        eval_steps = 100
        patience = 5
        min_delta = 1e-3
        steps = max_steps
        # 바닥 = 처음 학습률(1e-3)을 factor(0.2)로 세 번 낮춘 값. 바닥에 닿으면 더 낮아지지 않아 조기 종료가 진행된다
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.2, patience=2, threshold=min_delta, min_lr=1e-3 * 0.2**3)
        best, best_step, bad_evals = float("-inf"), 0, 0
        best_state = {
            "img_head": copy.deepcopy(img_head.state_dict()),
            "txt_head": copy.deepcopy(txt_head.state_dict()),
        }
    else:
        warmup = steps // 10
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda s: (s+1)/warmup if s<warmup else 0.5 * (1 + math.cos(math.pi*(s-warmup)/(steps-warmup)))
        )
    
    img_head.train()
    txt_head.train()
    train_steps = 0
    for step in range(steps):
        img_idx, cap_idx = next(it)
        imgs = img[img_idx] # (b,D)
        caps = txt[cap_idx] # (b,D)
        img_embs = torch.nn.functional.normalize(img_head(imgs),dim=-1)
        txt_embs = torch.nn.functional.normalize(txt_head(caps),dim=-1)
        loss = crit(img_embs, txt_embs, log_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if eval_loss:
            if step % eval_steps == 0:
                recall_1 = validate()
                print(f"step {step:4d} | loss {loss.item():.4f} | scale {log_t.exp().item():6.2f} | bias {b.item():6.2f} "
                  f"| lr {scheduler.get_last_lr()[0]:.2e} | val recall@1 {float(recall_1):.4f} | best {float(best):.4f}@{best_step}", file=sys.stderr)
                lr_before = optimizer.param_groups[0]["lr"]
                scheduler.step(recall_1)
                if optimizer.param_groups[0]["lr"] < lr_before:
                    bad_evals = 0
                    
                if recall_1 > best * (1 + min_delta):
                    best, best_step, bad_evals = recall_1, step, 0
                    best_state = {
                        "img_head": copy.deepcopy(img_head.state_dict()),
                        "txt_head": copy.deepcopy(txt_head.state_dict()),
                    }
                else:
                    bad_evals += 1
                    if bad_evals >= patience:
                        step = best_step
                        img_head.load_state_dict(best_state["img_head"])
                        txt_head.load_state_dict(best_state["txt_head"])
                        break
        else:
            scheduler.step()
            if (step * 100 // steps) % 10 == 0 or step == steps - 1:
                print(f"step {step:4d} | loss {loss.item():.4f} | scale {log_t.exp().item():6.2f} | bias {b.item():6.2f} "
                  f"| lr {scheduler.get_last_lr()[0]:.2e}", file=sys.stderr)
    
    img_head.eval()
    txt_head.eval()
    return img_head, txt_head, step
# ----------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["zeroshot", "padding", "retrieval", "train", "all"], default="all")
    part = parser.parse_args().part
    device = pick_device()
    model, processor = load_siglip(device)
    t, b = scale_and_bias(model)
    print(f"device: {device} | 사전학습된 온도 t = {t:.2f}, 편향 b = {b:.2f}")

    if part in ("zeroshot", "all"):
        # 1) zero-shot 분류 (STL-10 1000장)
        images, labels, classes = load_stl10(n=1000)
        img = cached("stl10_test1000_img", lambda: encode_images(model, processor, images, device))
        prompts = [f"a photo of a {c}." for c in classes]
        txt = encode_texts(model, processor, prompts, device)
        out = zero_shot(img, txt, t, b)
        print("zero-shot 정확도:", (out["pred"] == labels).float().mean().item())
    
    if part in ("padding", "all"):
        # 관측: 패딩 방식별 차이 (GUIDE 실행 2 "관측")
        # TODO(직접): encode_texts(..., padding=False / "max_length" / "longest")로 같은 문장의 출력 벡터를 만들고,
        # "max_length" 기준 코사인 유사도, zero-shot 점수, 1등 클래스를 비교한다. "longest"는 함께 묶는 문장을 바꿔 본다.
        images, labels, classes = load_stl10(n=1000)
        for padding in ("max_length", "longest", False):
            img = cached("stl10_test1000_img", lambda: encode_images(model, processor, images, device))
            if padding == "longest":
                prompts = [f"find a photo of a {c}." if random() > 0.5 else f"a photo of a {c}." for c in classes]
            else:
                prompts = [f"a photo of a {c}." for c in classes]

            txt = encode_texts(model, processor, prompts, device, padding=padding)
            out = zero_shot(img, txt, t, b)
            print(f"zero-shot(padding={padding}) 정확도:", (out["pred"] == labels).float().mean().item())

    if part in ("retrieval", "all"):
        print("recall_at_k")
        # 2) 이미지-텍스트 검색 (Flickr8k 시험 1000장 × 캡션 5000개)
        images, captions, cap2img = load_flickr8k("test")
        img = cached("flickr8k_test_img", lambda: encode_images(model, processor, images, device))
        txt = cached("flickr8k_test_txt", lambda: encode_texts(model, processor, captions, device))
        print_table([recall_at_k(img, txt, cap2img)])

    if part in ("train", "all"):
        # 3) softmax vs sigmoid: 얼린 출력 벡터 위의 작은 head, 배치 크기만 바꿔 비교
        tr_images, tr_captions, _ = load_flickr8k("train")
        tr_img = cached("flickr8k_train_img", lambda: encode_images(model, processor, tr_images, device))
        tr_txt = cached("flickr8k_train_txt", lambda: encode_texts(model, processor, tr_captions, device))
        te_images, te_captions, te_cap2img = load_flickr8k("test")
        te_img = cached("flickr8k_test_img", lambda: encode_images(model, processor, te_images, device))
        te_txt = cached("flickr8k_test_txt", lambda: encode_texts(model, processor, te_captions, device))
        va_images, va_captions, va_cap2img = load_flickr8k("validation")
        va_img = cached("flickr8k_validation_img", lambda: encode_images(model, processor, va_images, device))
        va_txt = cached("flickr8k_validation_txt", lambda: encode_texts(model, processor, va_captions, device))
        # TODO(직접): 손실 두 가지 × 배치 크기 여러 개로 train_heads()를 돌리고,
        # 학습한 head를 통과시킨 시험 벡터로 recall_at_k()를 재서 rows에 담는다.
        print("train-same steps")
        rows: list[dict] = []
        steps = 1000
        for batch_size in [16,64,256]:
            for loss in ["softmax", "sigmoid"]:
                img_head, txt_head, _ = train_heads(loss, tr_img, tr_txt, va_img, va_txt, va_cap2img, batch_size=batch_size, steps=steps, seed=randint(0,10000))
                with torch.inference_mode():
                    img = torch.nn.functional.normalize(img_head(te_img),dim=-1)
                    txt = torch.nn.functional.normalize(txt_head(te_txt),dim=-1)
                rows.append({"case":f"{loss}-{batch_size}"} | recall_at_k(img, txt, te_cap2img))
        print_table(rows)

        print("train-same epochs")
        rows: list[dict] = []
        for batch_size in [16,64,256]:
            steps = 6000 * 5 // batch_size
            for loss in ["softmax", "sigmoid"]:
                img_head, txt_head, _ = train_heads(loss, tr_img, tr_txt, va_img, va_txt, va_cap2img, batch_size=batch_size, steps=steps, seed=randint(0,10000))
                with torch.inference_mode():
                    img = torch.nn.functional.normalize(img_head(te_img),dim=-1)
                    txt = torch.nn.functional.normalize(txt_head(te_txt),dim=-1)
                rows.append({"case":f"{loss}-{batch_size}"} | recall_at_k(img, txt, te_cap2img))
        print_table(rows)

        print("train-till saturation")
        rows: list[dict] = []
        for batch_size in [16,64,256]:
            steps = -1
            for loss in ["softmax", "sigmoid"]:
                img_head, txt_head, train_steps = train_heads(loss, tr_img, tr_txt, va_img, va_txt, va_cap2img, batch_size=batch_size, steps=steps, seed=randint(0,10000))
                with torch.inference_mode():
                    img = torch.nn.functional.normalize(img_head(te_img),dim=-1)
                    txt = torch.nn.functional.normalize(txt_head(te_txt),dim=-1)
                rows.append({"case":f"{loss}-{batch_size}-{train_steps}"} | recall_at_k(img, txt, te_cap2img))
        print_table(rows)

if __name__ == "__main__":
    main()
