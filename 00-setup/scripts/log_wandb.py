"""이 머신의 점검 결과(results/*.json, *.png)를 W&B run 하나로 올린다 — Mac·클라우드 기록 경로 확인용.

uv run python scripts/log_wandb.py                       # 로그인된 계정으로 업로드
WANDB_MODE=offline uv run python scripts/log_wandb.py    # 로컬 wandb/ 에만 기록
"""

import json
import sys
from pathlib import Path

import wandb

RESULTS = Path(__file__).resolve().parents[1] / "results"


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, f"{key}/"))
        elif isinstance(v, (int, float, str, bool)) or v is None:
            out[key] = v
        else:
            out[key] = json.dumps(v)
    return out


def main() -> None:
    tag = "mac" if sys.platform == "darwin" else "linux"
    torch_files = sorted(RESULTS.glob("torch-mps.json" if tag == "mac" else "torch-cu*.json"))
    files = [*torch_files, RESULTS / f"mujoco-{tag}.json", RESULTS / f"pusht-{tag}.json"]
    missing = [f.name for f in files if not f.exists()]
    if not torch_files or missing:
        sys.exit(f"점검 결과가 없다: {missing or 'torch-*.json'} — check_*.py를 먼저 실행한다")

    reports = {f.stem.split("-")[0]: json.loads(f.read_text()) for f in files}
    device = reports["torch"]["device"]
    run = wandb.init(
        project="physical-ai-pratics",
        group="00-setup",
        job_type="env-check",
        name=f"env-check-{tag}-{device}",
        tags=[tag, device],
        config={"python": reports["torch"]["python"], "platform": reports["torch"]["platform"]},
    )
    for name, report in reports.items():
        run.summary.update(flatten(report, f"{name}/"))
    run.log({p.stem: wandb.Image(str(p)) for p in sorted(RESULTS.glob(f"*-{tag}.png"))})
    for f in [*files, *RESULTS.glob("nvidia-smi.csv")]:  # scp가 안 되는 Pod에서도 원본을 회수할 수 있게
        run.save(str(f), base_path=str(RESULTS), policy="now")
    run.finish()


if __name__ == "__main__":
    main()
