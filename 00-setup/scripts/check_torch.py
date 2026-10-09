"""PyTorch 가속기 점검: 장치, dtype 지원, MPS 폴백, matmul 처리량, 학습 1스텝의 CPU 일치.

Mac(MPS)과 클라우드(CUDA)에서 같은 스크립트를 돌려 results/torch-<device>.json 으로 비교한다.
    uv run python scripts/check_torch.py
"""

import copy
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

RESULTS = Path(__file__).resolve().parents[1] / "results"

# MPS 커널이 없는 연산 (torch 2.11 실측). 로드맵에서 실제로 마주칠 것들이다:
#   linalg.qr   → torch.pca_lowrank, nn.init.orthogonal_
#   linalg.eigh → 공분산 기반 PCA
MPS_MISSING_PROBES = {
    "linalg.qr": "torch.linalg.qr(x)",
    "linalg.eigh": "torch.linalg.eigh(x @ x.T)",
    "pca_lowrank": "torch.pca_lowrank(x, q=4)",
}


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def matmul_tflops(device: torch.device, dtype: torch.dtype, n: int = 4096, iters: int = 20) -> float:
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    for _ in range(3):
        a @ b
    sync(device)
    t = time.perf_counter()
    for _ in range(iters):
        a @ b
    sync(device)
    return 2 * n**3 * iters / (time.perf_counter() - t) / 1e12


def dtype_support(device: torch.device) -> dict[str, str]:
    out = {}
    for dtype in (torch.float16, torch.bfloat16, torch.float64):
        try:
            (torch.ones(4, dtype=dtype, device=device) * 2).sum().item()
            out[str(dtype)] = "ok"
        except (TypeError, RuntimeError) as e:  # MPS는 float64를 지원하지 않는다
            out[str(dtype)] = f"{type(e).__name__}: {str(e).splitlines()[0][:100]}"
    return out


def train_step_matches_cpu(device: torch.device) -> dict[str, float]:
    """같은 초기값의 MLP를 한 스텝 학습해 장치와 CPU의 파라미터 차이를 잰다."""
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(64, 256), torch.nn.GELU(), torch.nn.Linear(256, 8))
    x, y = torch.randn(32, 64), torch.randn(32, 8)
    results = {}
    for dev in (torch.device("cpu"), device):
        m = copy.deepcopy(model).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        loss = torch.nn.functional.mse_loss(m(x.to(dev)), y.to(dev))
        loss.backward()
        opt.step()
        results[dev.type] = (loss.item(), torch.cat([p.detach().cpu().flatten() for p in m.parameters()]))
    (loss_cpu, p_cpu), (loss_dev, p_dev) = results["cpu"], results[device.type]
    return {"loss_abs_diff": abs(loss_cpu - loss_dev), "param_max_abs_diff": (p_cpu - p_dev).abs().max().item()}


def probe_mps_fallback() -> dict[str, dict[str, str]]:
    """폴백 환경변수는 torch import 시점에 읽히므로 하위 프로세스로 켜고 끈 결과를 비교한다."""
    out = {}
    for name, expr in MPS_MISSING_PROBES.items():
        code = (
            "import time, torch; x = torch.randn(512, 64, device='mps'); t = time.perf_counter(); "
            f"r = {expr}; torch.mps.synchronize(); print(f'ok {{(time.perf_counter() - t) * 1e3:.1f}}ms')"
        )
        row = {}
        for flag in ("0", "1"):
            env = {**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": flag}
            p = subprocess.run(
                [sys.executable, "-W", "ignore", "-c", code], env=env, capture_output=True, text=True, check=False
            )
            last = (p.stdout or p.stderr).strip().splitlines()[-1]
            row[f"fallback={flag}"] = last if p.returncode == 0 else last.split(":")[0]
        out[name] = row
    return out


def main() -> None:
    device = pick_device()
    report: dict = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "device": device.type,
        "PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        report["gpu"] = {
            "name": props.name,
            "memory_gb": round(props.total_memory / 2**30, 1),
            "cuda": torch.version.cuda,
        }
    elif device.type == "mps":
        report["gpu"] = {"recommended_max_memory_gb": round(torch.mps.recommended_max_memory() / 2**30, 1)}

    report["dtype_support"] = dtype_support(device)
    report["matmul_tflops"] = {"cpu_fp32": round(matmul_tflops(torch.device("cpu"), torch.float32, iters=5), 2)}
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        if report["dtype_support"].get(str(dtype), "ok") == "ok" and device.type != "cpu":
            report["matmul_tflops"][f"{device.type}_{str(dtype).removeprefix('torch.')}"] = round(
                matmul_tflops(device, dtype), 2
            )
    report["train_step_vs_cpu"] = train_step_matches_cpu(device)
    if device.type == "mps":
        report["mps_fallback"] = probe_mps_fallback()

    print(json.dumps(report, indent=2, ensure_ascii=False))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"torch-{device.type}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
