#!/usr/bin/env bash
# 클라우드 GPU 인스턴스(Ubuntu + NVIDIA) 안에서 실행한다: 코드 동기화 → uv 환경 → 점검 3종 → W&B → 결과 묶음.
#
#   curl -LsSf https://raw.githubusercontent.com/personal-waityet-dev/physical-ai-pratics/main/00-setup/cloud/rehearsal.sh | bash
#
# 끝나면 Mac에서 결과를 회수하고 인스턴스를 종료한다 (00-setup/README.md의 "클라우드 리허설" 참고).
# 이후 프로젝트도 SPARSE_DIRS와 마지막 실행 단계만 바꿔 같은 흐름을 쓴다.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/personal-waityet-dev/physical-ai-pratics.git}"
BRANCH="${BRANCH:-main}"
WORKDIR="${WORKDIR:-$HOME/physical-ai-pratics}"
SPARSE_DIRS="${SPARSE_DIRS:-00-setup common}"  # docs/papers(90MB)는 받지 않는다

# 1. 시스템 패키지: MuJoCo EGL/OSMesa 렌더링, torchcodec이 쓰는 ffmpeg 공유 라이브러리
if command -v apt-get >/dev/null; then
  SUDO=$([ "$(id -u)" -eq 0 ] && echo "" || echo sudo)
  $SUDO apt-get update -qq
  DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq git curl libegl1 libgl1 libgles2 libosmesa6 ffmpeg >/dev/null
fi

# 2. uv
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# 3. 코드 (git으로만 동기화한다)
if [ -d "$WORKDIR/.git" ]; then
  git -C "$WORKDIR" pull --ff-only
else
  git clone --depth 1 --filter=blob:none --sparse -b "$BRANCH" "$REPO_URL" "$WORKDIR"
  # shellcheck disable=SC2086
  git -C "$WORKDIR" sparse-checkout set $SPARSE_DIRS
fi
cd "$WORKDIR/00-setup"

# 4. 환경: Mac에서 만든 uv.lock을 그대로 쓴다 (lock은 플랫폼 공통)
uv sync --locked

# 5. 점검
mkdir -p results
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee results/nvidia-smi.csv
uv run python scripts/check_torch.py
# 컨테이너에 NVIDIA EGL 라이브러리가 없으면(graphics capability 미포함) EGL이 실패한다 → CPU 렌더러로
MUJOCO_GL=egl uv run python scripts/check_mujoco.py --probe || MUJOCO_GL=osmesa uv run python scripts/check_mujoco.py --probe
SDL_VIDEODRIVER=dummy uv run python scripts/check_pusht.py

# 6. W&B: 제공자 secret으로 WANDB_API_KEY를 넣어 두었다면 결과를 Mac run과 같은 프로젝트에 올린다
if [ -n "${WANDB_API_KEY:-}" ]; then
  uv run python scripts/log_wandb.py
else
  echo "WANDB_API_KEY가 없어 W&B 기록은 건너뛴다"
fi

# 7. 결과 묶음
tar czf "$HOME/setup-results.tgz" -C "$WORKDIR/00-setup" results
echo
echo "완료. Mac의 저장소 루트에서 회수:"
echo "  scp -P <port> <user>@<host>:~/setup-results.tgz /tmp/ && tar xzf /tmp/setup-results.tgz -C 00-setup/"
echo "회수했으면 인스턴스를 종료(terminate)한다. stop만 하면 디스크 비용이 계속 나간다."
