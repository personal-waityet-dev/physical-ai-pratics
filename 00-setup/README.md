# 00-setup — 0주차 환경 준비

[ROADMAP §9.1](../ROADMAP.md#91-준비-체크리스트-0주차) 체크리스트의 실행 기록이다. 같은 점검 스크립트를 Mac과 클라우드 GPU에서 돌려 결과를 나란히 남긴다.

## 실행

```bash
cd 00-setup
uv sync                                          # Python 3.12 + lerobot[pusht] 0.6.1 + mujoco 3.15 (torch 2.11)
uv run python scripts/check_torch.py             # 장치·dtype·MPS 폴백·matmul 처리량·학습 1스텝 CPU 일치
uv run python scripts/check_mujoco.py --probe    # GL 백엔드, 접촉 시뮬레이션, 카메라 2대 RGB·depth
uv run python scripts/check_pusht.py             # lerobot/pusht 로드·디코딩, gym-pusht에서 데모 행동 재생
uv run python scripts/log_wandb.py               # 위 결과를 W&B run 하나로 업로드 (group 00-setup)
```

결과는 `results/`에 `*-mac.json`·`*-mps.json`(Mac), `*-linux.json`·`*-cuda.json`(클라우드)과 그림으로 저장된다.

**uv 규약** (이후 모든 프로젝트 공통): 프로젝트 디렉토리마다 `pyproject.toml` + `.python-version` + `uv.lock`을 두고 `uv run`으로만 실행한다. Python은 uv가 관리하는 인터프리터를 쓰고 시스템 Python 3.9는 건드리지 않는다. `uv.lock`은 플랫폼 공통이라 Mac에서 만든 lock을 클라우드에서 `uv sync --locked`로 그대로 재현한다.

## 체크리스트

| 항목 | 상태 | 근거 |
|---|---|---|
| uv로 프로젝트별 독립 환경 | 완료 | 이 디렉토리, Python 3.12.13 (uv 관리) |
| PyTorch MPS 동작·폴백 이해 | 완료 | [`results/torch-mps.json`](results/torch-mps.json) |
| MuJoCo 오프스크린 렌더링 | 완료 | [`results/mujoco-mac.json`](results/mujoco-mac.json), [그림](results/mujoco-render-mac.png) |
| LeRobot `lerobot/pusht` 로드·재생 | 완료 | [`results/pusht-mac.json`](results/pusht-mac.json), [그림](results/pusht-replay-mac.png) |
| 노트 템플릿 | 완료 | [`docs/notes/_template.md`](../docs/notes/_template.md) |
| 저장소 규약 파일 | 완료 | 루트 `.gitignore`(대용량 산출물·가상환경 제외), `.gitattributes`(PDF를 바이너리로) |
| 실험 기록 도구 결정 | 완료 | **W&B** — LeRobot 학습 스크립트가 W&B만 지원(`--wandb.enable=true`), Mac·클라우드 run을 한곳에 모음. 아래 "실험 기록 규약" |
| 클라우드 GPU 리허설 | **남음** | 스크립트 준비 완료 — 아래 "클라우드 리허설" |

## 결과 (Mac: M3 Pro 18GB, macOS 26.6)

| 항목 | Mac (MPS) | 클라우드 (RTX 4090) |
|---|---|---|
| matmul fp32 / bf16 (TFLOPS, 4096²) | 4.9 / 5.9 (CPU fp32 1.3) | 리허설 후 기입 |
| 가속기 메모리 상한 | 13.3GB (`recommended_max_memory`) | |
| 학습 1스텝 장치-CPU 파라미터 차이 | 1.1e-6 | |
| MuJoCo 렌더 fps (224² / 640×480) | 약 610 / 545 (`cgl`) | |
| PushT 비디오 디코딩 (96², 프레임당) | 1.7–1.9ms (`pyav`) | |
| PushT env step | 약 890 step/s | |

## 발견한 것 — 이후 프로젝트에 주는 함의

**PyTorch / MPS**
- **float64 미지원.** numpy 배열(기본 float64)을 그대로 `.to("mps")`하면 `TypeError`가 난다. 데이터 로더에서 `float32`로 변환하는 습관을 들인다.
- **MPS 커널이 없는 연산**(torch 2.11 실측): `linalg.qr`, `linalg.eigh`, `linalg.lstsq`, `linalg.eig`, `linalg.matrix_exp`, `ctc_loss`. 따라서 `torch.pca_lowrank`와 `nn.init.orthogonal_`도 MPS 텐서에서는 실패한다(실측). **P01의 특징 PCA 같은 분해 계산은 CPU로 옮겨서** 하거나 폴백을 켠다.
- `PYTORCH_ENABLE_MPS_FALLBACK=1`은 커널이 없는 연산만 CPU로 보내 실행한다(장치 간 복사 포함, 결과는 맞지만 느리다). **torch를 import하기 전에** 설정해야 효과가 있다. 기본은 꺼 두어서 무엇이 CPU로 가는지 에러로 알 수 있게 하고, 서드파티 정책 코드(LeRobot 등)를 돌릴 때만 켠다.
- **메모리 상한은 18GB가 아니라 13.3GB**다. §9.2에서 "빠듯함"으로 분류한 π0/π0.5 추론(≈14GB)은 이 상한을 넘으므로, 기본 계획을 클라우드로 둔다.
- bf16 이득이 작다(fp32 대비 약 1.2배). Mac에서는 정밀도를 낮춰 얻는 속도보다 모델·배치 크기를 줄이는 쪽이 효과가 크다.

**MuJoCo**
- macOS에서 `MUJOCO_GL`로 쓸 수 있는 값은 `cgl`·`glfw`뿐이다. `egl`·`osmesa`는 값 자체가 거부된다. 스크립트에서는 `os.environ.setdefault("MUJOCO_GL", "cgl" if darwin else "egl")`를 `import mujoco`보다 앞에 둔다.
- macOS OpenGL 4.1에는 `ARB_clip_control`이 없어서 depth 렌더 정밀도가 제한된다는 경고가 뜬다. P03에서 depth 카메라를 쓸 때 원거리 정밀도를 확인한다.

**LeRobot / PushT**
- **ffmpeg가 없으면 torchcodec이 로드되지 않고 LeRobot이 자동으로 `pyav`로 폴백**한다. 96² 영상은 pyav로 충분하다. LIBERO/ALOHA처럼 큰 영상에서 디코딩이 병목이 되면 `brew install ffmpeg`로 torchcodec을 살린다.
- **opencv 충돌.** gym-pusht는 `opencv-python`을, lerobot은 `opencv-python-headless`를 요구하는데, 둘이 같은 `cv2/` 디렉토리를 덮어쓴다. 그래서 `pyproject.toml`에서 `[tool.uv] exclude-dependencies = ["opencv-python"]`로 headless 하나만 남겼다. 이미 둘 다 설치된 환경에서 제외하면 공유 파일이 지워져 `cv2`가 깨지므로 `uv sync --reinstall-package opencv-python-headless`로 복구한다. PushT를 쓰는 프로젝트(P04 등)는 이 설정을 그대로 가져간다.
- 실행 시 objc "Class … is implemented in both" 경고가 나온다. opencv-headless 휠이 SDL2·ffmpeg dylib를 자체 포함해서 pygame·av와 중복 로드되기 때문이다. 지금은 무해하지만, 원인 모를 크래시가 나면 가장 먼저 의심한다.
- **PushT 행동 = 에이전트 목표 위치**(절대 픽셀 좌표 0–512, 10Hz, PD 추종). 에이전트는 KINEMATIC 강체라 블록에 밀리지 않는다. 그래서 데모 행동을 열린 루프로 재생하면 에이전트 궤적이 데이터셋 상태와 정확히 일치한다(오차 0.0px). 반면 데이터셋에 T 블록 포즈가 없어서 데모 전체는 재현할 수 없다. P03의 행동 공간 비교표에 넣을 첫 항목이다.
- **데이터셋의 성공 라벨**: 206개 에피소드 모두 `next.success`가 False이고, 에피소드별 최대 `next.reward`는 중앙값 0.896(범위 0.813–0.949)이다. env의 성공 기준은 커버리지 > 0.95다. **P04에서 성공률을 논문·LeRobot 보고치와 비교하기 전에 성공 판정 정의부터 확인**한다.

## 실험 기록 규약 (W&B)

- **로그인**: Mac에서는 `uvx wandb login`을 한 번 실행한다(키는 `~/.netrc`에 저장). 클라우드에서는 제공자의 secret/환경변수 기능으로 `WANDB_API_KEY`를 주입한다. 키는 저장소·스크립트·`.env`에 쓰지 않는다.
- **이름 규칙**: project `physical-ai-pratics`, group은 프로젝트 디렉토리(`01-vlm-from-llm` …), tags는 장치(`mps`/`cuda`)와 실험 축. LeRobot이라면 `--wandb.enable=true --wandb.project=physical-ai-pratics`.
- **항상 남기는 것**: 시드, 데이터·설정 해시(또는 git commit), 그리고 정책 평가라면 에피소드 수와 성공률 Wilson 95% 신뢰구간(ROADMAP §5 원칙 1).
- **오프라인**: 네트워크가 없거나 키를 넣기 싫은 환경에서는 `WANDB_MODE=offline`으로 기록하고 나중에 `wandb sync`로 올린다.
- `wandb`는 각 프로젝트의 `pyproject.toml`에 필요할 때 추가한다. 00-setup에는 기록 경로 점검용으로 들어 있다.

## 클라우드 리허설

목표는 "인스턴스 생성 → git으로 코드 동기화 → 스크립트 실행 → 결과 회수 → 인스턴스 종료"를 한 번 끝까지 해 보는 것이다. 예상 비용은 RTX 4090 30분, $0.5 미만이다.

1. **(직접)** 클라우드 계정을 만들고 결제 수단과 **지출 한도·알림**을 설정한다. 선불 크레딧 방식(RunPod, Vast.ai)이면 충전액이 사실상 상한이므로 $10 정도만 충전한다. 제공자의 secret 기능에 `WANDB_API_KEY`를 등록해 두면 이후 모든 인스턴스에 자동으로 주입된다(키는 https://wandb.ai/authorize).
2. 이 디렉토리의 변경을 GitHub `main`에 push한다. 스크립트가 GitHub에서 sparse clone한다.
3. RTX 4090 인스턴스를 띄운다(Ubuntu + PyTorch 템플릿, SSH 키 등록).
4. SSH로 접속해 실행한다.
   ```bash
   curl -LsSf https://raw.githubusercontent.com/personal-waityet-dev/physical-ai-pratics/main/00-setup/cloud/rehearsal.sh | bash
   ```
5. Mac의 저장소 루트에서 결과를 회수한다. 스크립트가 끝날 때 명령을 출력한다.
   ```bash
   scp -P <port> <user>@<host>:~/setup-results.tgz /tmp/ && tar xzf /tmp/setup-results.tgz -C 00-setup/
   ```
6. **인스턴스를 terminate하고, 콘솔에서 실행 중인 인스턴스가 0인지 확인한다.** stop만 하면 디스크 비용이 계속 나간다.
7. 위 결과표의 "클라우드" 열을 채운다.

[`cloud/rehearsal.sh`](cloud/rehearsal.sh)가 처리하는 일은 다음과 같다. apt로 EGL/OSMesa·ffmpeg 설치, uv 설치, `00-setup`과 `common`만 sparse clone(PDF 90MB 제외), `uv sync --locked`, 점검 3종, 그리고 `WANDB_API_KEY`가 있으면 W&B 업로드. MuJoCo는 EGL을 먼저 시도하고, 컨테이너에 NVIDIA EGL 라이브러리가 없으면 OSMesa(CPU 렌더)로 넘어간다. 이후 프로젝트도 `SPARSE_DIRS`와 실행 단계만 바꿔 같은 흐름을 쓴다.

## 남은 일

- [x] W&B 로그인 (2026-10-09, `~/.netrc`)
- [ ] 클라우드 리허설(위 절차) 후 결과표 클라우드 열 채우기
