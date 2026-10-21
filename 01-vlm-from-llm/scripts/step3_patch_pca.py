"""실행 3 — SigLIP vs DINOv2 패치 특징을 PCA로 색칠해 비교.

    uv run python scripts/step3_patch_pca.py                  # 전부
    uv run python scripts/step3_patch_pca.py --part tokens    # tokens | pca 중 하나
    uv run python scripts/step3_patch_pca.py --n-images 8 --size 224

보조 코드(이미지 불러오기, 두 모델에 같은 픽셀을 넣는 전처리, 모델 출력 계산, 그림 저장)는 준비돼 있다.
`TODO(직접)` 표시가 있는 함수와 main()의 각 부분을 채운다. 예측·결과는 README "실행 기록"에 쓴다.

모델(처음 한 번 내려받음): SigLIP은 실행 1·2와 같은 체크포인트(캐시 있음), DINOv2-small 약 90MB.
이미지는 실행 1의 data/cats.jpg와 Flickr8k 시험 분할의 앞쪽 이미지를 쓴다(실행 2에서 내려받은 캐시).
그림은 outputs/step3/에 저장한다(저장소 .gitignore 대상).
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import transformers
from common import pick_device, print_table
from datasets import load_dataset
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, SiglipVisionModel

transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()

HERE = Path(__file__).resolve().parents[1]
DATA = HERE / "data"
OUTPUTS = HERE / "outputs" / "step3"  # 저장소 .gitignore 대상

SIGLIP = "google/siglip-so400m-patch14-224"  # 실행 1·2와 같은 체크포인트
DINOV2 = "facebook/dinov2-small"  # GUIDE 실행 3
FLICKR8K = "jxie/flickr8k"
PATCH = 14  # 두 모델 모두 패치 크기 14

MPS_MEMORY_FRACTION = 0.6  # 00-setup/README.md "Mac 자원 안전 규약"


# ---------------------------------------------------------------- 보조 코드


def load_images(n: int) -> list[tuple[str, Image.Image]]:
    """비교할 이미지 n장. 고르지 않고 정해진 순서로 뽑는다(cats.jpg, 그다음 Flickr8k 시험 분할 앞에서부터).

    반환: [(이름, PIL 이미지), ...]
    """
    out = []
    cats = DATA / "cats.jpg"  # 실행 1에서 내려받은 COCO val2017 000000039769
    if cats.exists():
        out.append(("cats", Image.open(cats).convert("RGB")))
    ds = load_dataset(FLICKR8K, data_files={"test": "data/test-*.parquet"}, split="test")
    for i in range(n - len(out)):
        out.append((f"flickr8k_test_{i}", ds[i]["image"].convert("RGB")))
    return out[:n]


def square_resize(image: Image.Image, size: int) -> Image.Image:
    """가운데를 정사각형으로 잘라 size×size로 줄인다.

    두 모델의 처리기는 기본 전처리가 다르다(SigLIP: 비율 무시하고 224×224로 늘림, DINOv2: 짧은 변 256으로 줄인 뒤
    가운데 224 자르기). 그대로 쓰면 두 모델이 보는 영역이 달라 그림을 나란히 비교할 수 없으므로,
    여기서 같은 size×size 이미지를 만든 뒤 정규화만 모델별로 다르게 한다(to_pixel_values).
    """
    w, h = image.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return image.crop((left, top, left + s, top + s)).resize((size, size), Image.BICUBIC)


def to_pixel_values(image: Image.Image, mean, std) -> torch.Tensor:
    """size×size PIL 이미지 → (1, 3, size, size) 텐서. [0, 1]로 바꾼 뒤 모델별 평균·표준편차로 정규화한다."""
    x = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).view(image.height, image.width, 3)
    x = x.permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return ((x - mean) / std).unsqueeze(0)


def load_encoders(device: torch.device) -> dict:
    """두 비전 인코더를 float32로 불러온다. 반환: {"siglip": (model, mean, std), "dinov2": (model, mean, std)}

    SigLIP은 이미지 인코더만(SiglipVisionModel), DINOv2는 Dinov2Model(AutoModel)이다.
    처리기는 정규화 값(image_mean, image_std)만 가져다 쓴다.
    """
    out = {}
    for key, name, cls in [("siglip", SIGLIP, SiglipVisionModel), ("dinov2", DINOV2, AutoModel)]:
        model = cls.from_pretrained(name).to(device).eval()
        proc = AutoImageProcessor.from_pretrained(name)
        out[key] = (model, proc.image_mean, proc.image_std)
    return out


@torch.inference_mode()
def run_encoder(key: str, model, pixel_values: torch.Tensor, device: torch.device):
    """모델 출력 객체(HF ModelOutput)를 그대로 돌려준다. 어느 값을 패치 특징으로 쓸지는 patch_features()에서 정한다.

    size가 224가 아니면 SigLIP은 위치 표를 보간한다(실행 1 Q3). DINOv2는 입력 크기에 맞춰 늘 보간한다.
    """
    pixel_values = pixel_values.to(device)
    if key == "siglip":
        return model(pixel_values=pixel_values, interpolate_pos_encoding=pixel_values.shape[-1] != 224)
    return model(pixel_values=pixel_values)


def save_grid(
    names: list[str], images: list[Image.Image], panels: dict[str, list[torch.Tensor]], path: Path, title: str = ""
) -> None:
    """한 행에 이미지 하나: 원본 | 패널 1 | 패널 2 ... 를 그려 path에 저장한다.

    panels: {"열 제목": [이미지별 (h, w, 3) 텐서, 값은 [0, 1]], ...}. 패치 격자를 픽셀 크기로 늘리지 않고
    격자 그대로 그린다(nearest). 그래서 칸 하나가 패치 하나다.
    """
    cols = 1 + len(panels)
    fig, axes = plt.subplots(len(images), cols, figsize=(3 * cols, 3 * len(images)), squeeze=False)
    for r, (name, image) in enumerate(zip(names, images)):
        axes[r][0].imshow(image)
        axes[r][0].set_ylabel(name, fontsize=8)
        for c, (col, maps) in enumerate(panels.items(), start=1):
            axes[r][c].imshow(maps[r].clamp(0, 1).cpu().numpy(), interpolation="nearest")
            if r == 0:
                axes[r][c].set_title(col, fontsize=9)
        if r == 0:
            axes[r][0].set_title("image", fontsize=9)
        for ax in axes[r]:
            ax.set_xticks([])
            ax.set_yticks([])
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"저장: {path.relative_to(HERE)}")


# ---------------------------------------------------------------- 직접 구현


def patch_features(key: str, output, size: int) -> torch.Tensor:
    """모델 출력에서 패치 특징만 꺼내 (h, w, D) 격자로 만든다. h = w = size // PATCH

    TODO(직접): 출력 객체에서 어느 값을 쓸지 정하고(예측 Q2), 패치가 아닌 토큰이 있으면 덜어낸다(예측 Q3).
    토큰 순서가 격자의 어떤 순서(행 우선/열 우선)인지도 확인한다.
    """
    raise NotImplementedError


def pca_rgb(feats: list[torch.Tensor], n_components: int = 3) -> tuple[list[torch.Tensor], torch.Tensor]:
    """패치 특징 격자들 [(h, w, D), ...]을 주성분 3개로 사영해 RGB 그림 [(h, w, 3), 값 [0, 1]]으로 만든다.

    TODO(직접): 정할 것
    - PCA를 이미지마다 따로 맞출지, 모든 이미지의 패치를 모아 한 번에 맞출지(색의 뜻이 이미지 간에 같아지는가)
    - 평균을 빼는지(중심화), 특징을 정규화하는지
    - 사영값을 [0, 1]로 옮기는 방법(성분별 최솟값·최댓값, 백분위 등)
    반환: (RGB 그림 목록, 주성분 n_components개의 설명 분산 비율 (n_components,))
    """
    raise NotImplementedError


# ----------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["tokens", "pca", "all"], default="all")
    parser.add_argument("--n-images", type=int, default=6)
    parser.add_argument("--size", type=int, default=224, help="입력 한 변 픽셀 수. 14의 배수")
    args = parser.parse_args()
    assert args.size % PATCH == 0, "size는 14의 배수여야 한다"

    device = pick_device()
    if device.type == "mps":
        torch.mps.set_per_process_memory_fraction(MPS_MEMORY_FRACTION)
    encoders = load_encoders(device)
    named = load_images(args.n_images)
    names = [n for n, _ in named]
    images = [square_resize(im, args.size) for _, im in named]
    print(f"device: {device} | 이미지 {len(images)}장, {args.size}×{args.size}")

    # 이미지마다 두 모델의 출력 객체. 키: "siglip" | "dinov2"
    outputs = {
        key: [run_encoder(key, model, to_pixel_values(im, mean, std), device) for im in images]
        for key, (model, mean, std) in encoders.items()
    }

    if part_on(args.part, "tokens"):
        # 1) 출력 모양 확인 (예측 Q2·Q3)
        # TODO(직접): 각 모델 출력 객체에 어떤 값들이 있고 모양이 무엇인지 출력한다.
        # 패치 수(손 계산)와 출력 토큰 수를 rows에 함께 담아 비교한다.
        rows: list[dict] = []
        print_table(rows)

    if part_on(args.part, "pca"):
        # 2) 패치 특징 PCA 색칠 (예측 Q1)
        # TODO(직접): patch_features()로 격자를 만들고 pca_rgb()로 색칠해 save_grid()로 저장한다.
        # 두 모델의 설명 분산 비율도 rows에 담는다.
        rows = []
        print_table(rows)


def part_on(part: str, name: str) -> bool:
    return part in (name, "all")


if __name__ == "__main__":
    main()
