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
    raise NotImplementedError


def recall_at_k(img: torch.Tensor, txt: torch.Tensor, caption_to_image: torch.Tensor, ks=(1, 5, 10)) -> dict:
    """이미지-텍스트 검색의 recall@k. img (N, D), txt (5N, D), caption_to_image (5N,)

    TODO(직접): 이미지→텍스트(정답 캡션 5개 중 하나라도 상위 k개에 들면 성공)와
    텍스트→이미지(정답 이미지가 상위 k개에 들면 성공)를 각각 계산한다.
    반환: {"i2t@1": float, "i2t@5": ..., "t2i@1": ..., ..., "n_images": N, "n_captions": 5N}
    """
    raise NotImplementedError


class ProjectionHead(torch.nn.Module):
    """얼린 출력 벡터 위에 학습하는 작은 사영. 이미지와 텍스트 각각에 하나씩 둔다.

    TODO(직접): 구조(선형 한 층 등), 출력 차원, 출력 정규화 여부를 정한다.
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def softmax_loss(img: torch.Tensor, txt: torch.Tensor, log_t: torch.Tensor) -> torch.Tensor:
    """CLIP식 대조 손실. img·txt (B, d)는 정규화된 벡터이고 같은 행 번호끼리 정답 쌍.

    TODO(직접): 양방향 교차 엔트로피.
    """
    raise NotImplementedError


def sigmoid_loss(img: torch.Tensor, txt: torch.Tensor, log_t: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """SigLIP식 쌍별 시그모이드 손실. img·txt (B, d)는 정규화된 벡터이고 같은 행 번호끼리 정답 쌍.

    TODO(직접): 모든 쌍에 대한 이진 분류 손실. 배치 크기로 어떻게 나눌지도 정한다.
    """
    raise NotImplementedError


def train_heads(loss_name: str, img: torch.Tensor, txt: torch.Tensor, batch_size: int, steps: int, seed: int = 0):
    """얼린 출력 벡터 위에서 ProjectionHead 두 개(이미지·텍스트)를 학습한다.

    img (N, D)는 학습 이미지, txt (5N, D)는 학습 캡션. 미니배치는 pair_batches()로 만든다.
    TODO(직접): 온도·편향을 학습 가능하게 둘지와 초기값, 옵티마이저와 학습률을 정한다.
    두 손실의 비교는 같은 데이터·같은 head·같은 스텝 수에서 배치 크기만 바꾼다(GUIDE 검증 기준).
    반환: (이미지 head, 텍스트 head)
    """
    raise NotImplementedError


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
        pass

    if part in ("retrieval", "all"):
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
        # TODO(직접): 손실 두 가지 × 배치 크기 여러 개로 train_heads()를 돌리고,
        # 학습한 head를 통과시킨 시험 벡터로 recall_at_k()를 재서 rows에 담는다.
        rows: list[dict] = []
        print_table(rows)


if __name__ == "__main__":
    main()
