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


def memory_mb(device: torch.device) -> float:
    """현재 장치에 할당된 텐서 메모리(MB). 최댓값이 아니라 호출 시점의 값이다."""
    if device.type == "cuda":
        return torch.cuda.memory_allocated() / 2**20
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / 2**20
    return float("nan")


def print_table(rows: list[dict]) -> None:
    if not rows:
        print("(측정 결과 없음)")
        return
    keys = list(rows[0])
    print(" | ".join(keys))
    for r in rows:
        print(" | ".join(str(r[k]) for k in keys))


# ---------------------------------------------------------------- 직접 구현


def patchify(pixel_values: torch.Tensor, patch_size: int) -> torch.Tensor:
    """(B, C, H, W) 이미지를 (B, N, C·P·P) 패치 벡터 수열로 바꾼다.

    TODO(직접): reshape/permute만 쓴다(합성곱, unfold 금지).
    - 패치 벡터 안의 값 순서가 합성곱 가중치 (D, C, P, P)를 펼친 순서와 맞아야 가중치를 그대로 쓸 수 있다.
    - 패치들의 순서(N 축)는 HF 출력을 수열로 펼친 순서와 맞춘다.
    """
    raise NotImplementedError


def patch_embed(patches: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """(B, N, C·P·P) 패치 벡터에 패치 임베딩 층의 가중치를 적용해 (B, N, D)를 만든다.

    TODO(직접): weight는 합성곱 가중치 모양 그대로 (D, C, P, P), bias는 (D,)다.
    """
    raise NotImplementedError


def compare(mine: torch.Tensor, hf_conv_out: torch.Tensor) -> dict:
    """직접 만든 (B, N, D)와 HF 합성곱 출력을 비교한다.

    TODO(직접): HF 출력을 (B, N, D)로 바꾼 뒤 최대 절대 오차와 allclose 결과를 돌려준다.
    허용 오차는 장치와 dtype을 근거로 정하고 README에 적는다.
    """
    raise NotImplementedError


# ----------------------------------------------------------------


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

    # 2) 해상도별 토큰 수·메모리·지연
    # TODO(직접): 224/384/512 각각 모델 전체 forward를 timed()로 재고, 토큰 수·메모리·지연을 rows에 담는다.
    # - 측정 조건(장치, dtype, 배치, 워밍업 횟수)도 함께 남긴다.
    # - 512는 그대로 넣으면 어떻게 되는지부터 확인하고(예측 Q3), 나온 에러를 README에 기록한다.
    # - 384 예측(Q2)은 MODEL_384의 위치 표 크기와 실제 출력 토큰 수로 확인한다.
    rows: list[dict] = []
    print_table(rows)


if __name__ == "__main__":
    main()
