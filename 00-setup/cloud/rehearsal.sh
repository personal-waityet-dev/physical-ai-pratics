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

# 2. uv — 템플릿에 든 uv는 오래됐을 수 있다(RunPod PyTorch 템플릿은 0.9.0이라 [tool.uv] exclude-dependencies를 못 읽는다).
#    항상 최신을 ~/.local/bin에 설치하고 PATH 앞에 둔다.
curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
export PATH="$HOME/.local/bin:$PATH"
uv --version

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

# 6. W&B: 키가 있으면 결과를 Mac run과 같은 프로젝트에 올린다.
#    키를 찾는 순서: WANDB_API_KEY → RunPod secret(RUNPOD_SECRET_wandb_api_key).
#    RunPod 템플릿은 컨테이너 환경변수를 /etc/rp_environment에 남겨 SSH 세션이 읽게 하므로 그것도 읽는다.
#    이 파일은 PATH도 export하므로 그대로 source하면 2단계에서 앞에 둔 최신 uv가 밀려난다 → 서브셸에서 키만 꺼낸다.
if [ -z "${WANDB_API_KEY:-}" ] && [ -f /etc/rp_environment ]; then
  # shellcheck disable=SC1091
  WANDB_API_KEY="$(
    set +eu
    . /etc/rp_environment >/dev/null 2>&1
    printf '%s' "${WANDB_API_KEY:-${RUNPOD_SECRET_wandb_api_key:-${RUNPOD_SECRET_WANDB_API_KEY:-}}}"
  )" || true
fi
WANDB_API_KEY="${WANDB_API_KEY:-${RUNPOD_SECRET_wandb_api_key:-${RUNPOD_SECRET_WANDB_API_KEY:-}}}"
if [ -n "$WANDB_API_KEY" ]; then
  export WANDB_API_KEY
  uv run python scripts/log_wandb.py
else
  echo "W&B 키가 없어 기록은 건너뛴다 (00-setup/README.md '클라우드 리허설' 참고)"
fi

# 7. 결과 묶음
tar czf "$HOME/setup-results.tgz" -C "$WORKDIR/00-setup" results
echo
echo "완료. Mac의 저장소 루트에서 회수:"
echo "  scp -P <port> <user>@<host>:~/setup-results.tgz /tmp/ && tar xzf /tmp/setup-results.tgz -C 00-setup/"
echo "회수했으면 인스턴스를 종료(terminate)한다. stop만 하면 디스크 비용이 계속 나간다."
