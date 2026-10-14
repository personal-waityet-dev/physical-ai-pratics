"""실행 1 — 패치화: reshape로 만든 패치 임베딩이 HF SigLIP과 같은지 확인하고, 해상도별 비용을 잰다.

    uv run python scripts/step1_patchify.py

보조 코드(장치, 모델·이미지 불러오기, 측정 도구)는 준비돼 있다.
`TODO(직접)` 표시가 있는 함수 세 개와 main()의 측정 부분을 채운다. 예측과 결과는 README "실행 기록"에 쓴다.
"""

import statistics
import time
import urllib.request
from pathlib import Path

import torch
import transformers
from PIL import Image
from transformers import AutoImageProcessor, SiglipVisionModel

# 비전 인코더만 불러오면 텍스트 쪽 가중치를 건너뛴다는 보고서가 길게 찍힌다 — 정상이므로 끈다
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()

HERE = Path(__file__).resolve().parents[1]
DATA = HERE / "data"  # 저장소 .gitignore 대상
RESULTS = HERE / "results"

MODEL_224 = "google/siglip-so400m-patch14-224"  # 약 3.5GB, 처음 한 번 내려받는다
MODEL_384 = "google/siglip-so400m-patch14-384"  # 약 3.5GB, 384 모델의 위치 표를 확인할 때만
SAMPLE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"  # HF 문서에 자주 나오는 고양이 두 마리 사진


# ---------------------------------------------------------------- 보조 코드


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device: torch.device) -> None:
    """GPU 연산은 비동기라 시간을 재기 전에 끝날 때까지 기다린다."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def load_sample_image() -> Image.Image:
    path = DATA / "cats.jpg"
    if not path.exists():
        DATA.mkdir(exist_ok=True)
        urllib.request.urlretrieve(SAMPLE_URL, path)
    return Image.open(path).convert("RGB")


def load_model(name: str, device: torch.device):
    """비전 인코더만 float32로 불러온다. 반환: (model, image_processor)"""
    model = SiglipVisionModel.from_pretrained(name).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(name)
    return model, processor


def to_pixel_values(processor, image: Image.Image, size: int) -> torch.Tensor:
    """이미지를 size×size로 맞추고 정규화한 (1, 3, size, size) 텐서. 처리기 기본값은 모델의 학습 해상도다."""
    return processor(images=image, size={"height": size, "width": size}, return_tensors="pt")["pixel_values"]


@torch.inference_mode()
def hf_patch_conv(model, pixel_values: torch.Tensor) -> torch.Tensor:
    """HF 모델의 패치 임베딩 층만 통과시킨 결과(합성곱 출력 모양 그대로). 비교용 정답."""
    return model.embeddings.patch_embedding(pixel_values)


@torch.inference_mode()
def timed(fn, device: torch.device, warmup: int = 2, repeat: int = 5):
    """fn()을 warmup번 버리고 repeat번 재서 (마지막 결과, 지연 중앙값 ms)를 돌려준다."""
    for _ in range(warmup):
        fn()
    sync(device)
    times = []
    for _ in range(repeat):
        t = time.perf_counter()
        out = fn()
        sync(device)
        times.append((time.perf_counter() - t) * 1e3)
    return out, statistics.median(times)

@torch.inference_mode()
def peak_memory_mb(model: torch.nn.Module, fn, device: torch.device):
    """fn()을 한 번 실행하는 동안 늘어난 메모리의 최대값(MB)을 잰다. 반환: (fn의 결과, 최대 증가량 MB)

    실행 전부터 있던 메모리(가중치, 입력)는 빼고 잰다. 해상도별 활성값 비교용.
        _, mb = peak_memory_mb(model, lambda: model(pixel_values=pixel_values), device)

    - CUDA: 할당기가 기록하는 최대값으로 정확히 잰다.
    - MPS: 최대값 기록이 없어서, model의 모든 모듈이 끝날 때마다 현재 할당량을 적어 두고 그중 최대를 쓴다.
      모듈 안에서 잠깐 생겼다 사라지는 메모리는 놓치므로 실제보다 작게 나올 수 있다(하한값).
    - CPU: torch가 할당량을 알려주지 않아 nan.
    갈고리가 실행 시간을 늘리므로 timed()와 따로, 한 번만 잰다.
    """
    sync(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        out = fn()
        sync(device)
        return out, (torch.cuda.max_memory_allocated() - base) / 2**20
    if device.type == "mps":
        torch.mps.empty_cache()
        base = torch.mps.current_allocated_memory()
        samples = [base]
        hooks = [
            m.register_forward_hook(lambda *_: samples.append(torch.mps.current_allocated_memory()))
            for m in model.modules()
        ]
        try:
            out = fn()
            sync(device)
        finally:
            for h in hooks:
                h.remove()
        return out, (max(samples) - base) / 2**20
    return fn(), float("nan")


def print_table(rows: list[dict]) -> None:
    if not rows:
        print("(측정 결과 없음)")
        return
    keys = list(rows[0])
    print(" | ".join(keys))
    for r in rows:
        print(" | ".join(str(r.get(k, "")) for k in keys))


# ---------------------------------------------------------------- 직접 구현


def patchify(pixel_values: torch.Tensor, patch_size: int) -> torch.Tensor:
    """(B, C, H, W) 이미지를 (B, N, C·P·P) 패치 벡터 수열로 바꾼다.

    TODO(직접): reshape/permute만 쓴다(합성곱, unfold 금지).
    - 패치 벡터 안의 값 순서가 합성곱 가중치 (D, C, P, P)를 펼친 순서와 맞아야 가중치를 그대로 쓸 수 있다.
    - 패치들의 순서(N 축)는 HF 출력을 수열로 펼친 순서와 맞춘다.
    """
    (B, C, H, W) = pixel_values.size()
    patches = torch.reshape(pixel_values, (B, C, H//patch_size, patch_size, W//patch_size, patch_size)).permute(0,1,2,4,3,5).reshape(B,C,-1,patch_size,patch_size).permute(0,2,1,3,4)
    return patches


def patch_embed(patches: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """(B, N, C·P·P) 패치 벡터에 패치 임베딩 층의 가중치를 적용해 (B, N, D)를 만든다.

    TODO(직접): weight는 합성곱 가중치 모양 그대로 (D, C, P, P), bias는 (D,)다.
    """
    return torch.einsum('bncij,dcij->bnd',patches, weight) + bias

def compare(mine: torch.Tensor, hf_conv_out: torch.Tensor) -> dict:
    """직접 만든 (B, N, D)와 HF 합성곱 출력을 비교한다.

    TODO(직접): HF 출력을 (B, N, D)로 바꾼 뒤 아래 딕셔너리를 돌려준다.
    허용 오차는 장치와 dtype을 근거로 정하고 README에 적는다.

    반환값:
        {
            "shape_mine": tuple,    # mine.shape, 예: (1, N, D)
            "shape_hf": tuple,      # HF 출력을 (B, N, D)로 바꾼 뒤의 모양 — shape_mine과 같아야 한다
            "max_abs_err": float,   # (mine - hf).abs().max()
            "mean_abs_err": float,  # (mine - hf).abs().mean()
            "atol": float,          # torch.allclose에 넘긴 절대 허용 오차
            "rtol": float,          # torch.allclose에 넘긴 상대 허용 오차
            "allclose": bool,       # torch.allclose(mine, hf, atol=atol, rtol=rtol)
        }
    """
    """
    hf_conv_out : (B, D, P, P)
    """
    TOLERANCE_MAP = {
        ("cpu", torch.float64): (1e-08, 1e-10),
        ("cuda", torch.float64): (1e-07, 1e-09),
        ("mps", torch.float64): (1e-07, 1e-09),

        ("cpu", torch.float32): (1e-05, 1e-08),
        ("cuda", torch.float32): (1e-04, 1e-05),
        ("mps", torch.float32): (1e-04, 1e-05),

        ("cpu", torch.float16): (1e-03, 1e-03),
        ("cuda", torch.float16): (1e-03, 1e-03),
        ("mps", torch.float16): (1e-03, 1e-03),

        ("cpu", torch.bfloat16): (1e-02, 1e-03),
        ("cuda", torch.bfloat16): (1e-02, 1e-03),
        ("mps", torch.bfloat16): (1e-02, 1e-03),
    }
    def get_tolerance(tensor: torch.Tensor, fallback_rtol=1e-05, fallback_atol=1e-08):
        if not tensor.is_floating_point():
            return (0.0, 0.0)

        device_type = tensor.device.type
        dtype = tensor.dtype

        return TOLERANCE_MAP.get((device_type, dtype), (fallback_rtol, fallback_atol))

    (B, D, P, P) = hf_conv_out.size()
    hf_conv = hf_conv_out.permute(0, 2, 3, 1).reshape(B, -1, D)
    rtol, atol= get_tolerance(mine)
    return {
        "shape_mine" : mine.size(),
        "shape_hf" : hf_conv.size(),
        "max_abs_err" : (mine-hf_conv).abs().max(),
        "mean_abs_err" : (mine-hf_conv).abs().mean(),
        "atol": atol,
        "rtol": rtol,
        "allclose" : torch.allclose(mine,hf_conv,atol=atol,rtol=rtol)
    }


# ----------------------------------------------------------------


def check_embedding(model):
    emb = model.embeddings
    print("patch_embedding:", emb.patch_embedding)
    print("position_embedding:", emb.position_embedding)
 
def forward_model(model, pixel_values):
    output = model.forward(pixel_values=pixel_values,interpolate_pos_encoding=True)
    return {
        "hidden_state_shape": output.last_hidden_state.size(),
    }

def run_stats(model_size, device, image):
    warmup = 2
    repeat = 5
    model_string = MODEL_384 if model_size == 384 else MODEL_224
    model, processor = load_model(model_string, device)
    pixel_values=to_pixel_values(processor, image, model_size).to(device)
    return_dict = {
        "model" : model_string,
        "device":device.type,
        "dtype":pixel_values.dtype,
        "warmup":warmup,
        "repeat":repeat,
        "error" : "",
    }
    try:
        out, t = timed(lambda: forward_model(model,pixel_values), device, warmup, repeat)
        return_dict.update({**out, "timed": t})
        return_dict["memory"] = peak_memory_mb(model, lambda: forward_model(model,pixel_values), device)
        return return_dict
    except Exception as e:
        print(e)
        return_dict["error"] = str(e)
        return return_dict

def main() -> None:
    device = pick_device()
    image = load_sample_image()
    model, processor = load_model(MODEL_224, device)
    emb = model.embeddings
    print("device:", device, "| image:", image.size)
    print("patch_embedding:", emb.patch_embedding)
    print("position_embedding:", emb.position_embedding)
    # ↑ 노트의 결론(입력 C, 출력 D, 창 P, 이동 간격 P)과 예측 Q1·Q3의 위치 표 크기와 비교한다.

    # 1) 224px에서 직접 만든 패치 임베딩 == HF 패치 임베딩
    pixel_values = to_pixel_values(processor, image, 224).to(device)
    patches = patchify(pixel_values, emb.patch_size)
    mine = patch_embed(patches, emb.patch_embedding.weight, emb.patch_embedding.bias)
    print(compare(mine, hf_patch_conv(model, pixel_values)))

    # - 384 예측(Q2)은 MODEL_384의 위치 표 크기와 실제 출력 토큰 수로 확인한다.
    print("384 embedding")
    model, processor = load_model(MODEL_384, device)
    check_embedding(model)
    # 2) 해상도별 토큰 수·메모리·지연
    # TODO(직접): 224/384/512 각각 모델 전체 forward를 timed()로 재고, 토큰 수·메모리·지연을 rows에 담는다.
    # - 측정 조건(장치, dtype, 배치, 워밍업 횟수)도 함께 남긴다.
    # - 512는 그대로 넣으면 어떻게 되는지부터 확인하고(예측 Q3), 나온 에러를 README에 기록한다.
    model_size = [224, 384, 512]
    device_list = [torch.device("cpu")]
    if torch.cuda.is_available():
        device_list.append(torch.device("cuda"))
    if torch.backends.mps.is_available():
        device_list.append(torch.device("mps"))
    rows = [ run_stats(m,dv,image) for m in model_size for dv in device_list ]
    print_table(rows)


if __name__ == "__main__":
    main()
