"""P01 실행들이 함께 쓰는 보조 코드: 장치 선택, 동기화, 시간·메모리 측정, 표 출력.

실행 1(step1_patchify.py)에서 쓰던 보조 함수를 실행 2부터 함께 쓰려고 옮겨 왔다.
"""

import statistics
import time

import torch


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
    """fn()을 한 번 실행하는 동안 늘어난 메모리의 최대값(MB). 반환: (fn의 결과, 최대 증가량 MB)

    CUDA는 할당기의 최대값 기록으로 정확히, MPS는 모듈이 끝날 때마다 현재 할당량을 적어 그중 최대(하한값),
    CPU는 nan. 갈고리가 실행 시간을 늘리므로 timed()와 따로 한 번만 잰다.
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
        print("(결과 없음)")
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    print(" | ".join(keys))
    for r in rows:
        print(" | ".join(_fmt(r.get(k, "")) for k in keys))


def _fmt(v) -> str:
    return f"{v:.4g}" if isinstance(v, float) else str(v)
