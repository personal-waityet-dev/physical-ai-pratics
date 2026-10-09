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

1. **(직접) RunPod 계정과 지출 상한** — 선불 크레딧 + 자동 충전 끔 = 충전액이 곧 상한이다.
   - [console.runpod.io/signup](https://www.console.runpod.io/signup)에서 가입하고 이메일 인증, 2단계 인증을 켠다.
   - Billing에서 카드를 등록하고 **$10만 충전**한다. **Auto-pay는 켜지 않는다**(켜면 잔액이 기준 아래로 내려갈 때마다 카드에서 자동 충전돼 상한이 사라진다).
   - Billing에서 **잔액 부족 알림**(예: $3)을 설정한다.
   - 기본 "spend limit $80/시간"은 시간당 사용 속도 제한일 뿐 총액 상한이 아니다.
   - 잔액이 $0이 되면 실행 중인 Pod가 자동으로 멈춘다. network volume이 없는 Pod는 **terminate되어 데이터가 사라진다**. 리허설은 잃을 데이터가 없지만, 이후 프로젝트의 체크포인트는 Pod에만 두지 않는다(W&B artifact·HF Hub·rsync로 회수).
   - 요금 구조:
     - 연산은 초 단위로 과금된다.
     - container disk는 stop하면 과금되지 않는다.
     - volume disk는 stop한 동안 $0.20/GB/월이다(실행 중 $0.10).
     - network volume은 항상 $0.07/GB/월이다.
     - 그래서 끝낼 때는 stop이 아니라 terminate한다.
2. **(직접) SSH 키 등록** — 한 번만 하면 이후 모든 Pod에 자동으로 들어간다.
   ```bash
   ssh-keygen -t ed25519 -C "mac-m3pro"
   ```
   ```bash
   pbcopy < ~/.ssh/id_ed25519.pub
   ```
   콘솔 Credentials 페이지의 SSH Public Keys 탭에서 Add SSH Key로 붙여넣는다.
3. **(직접) W&B 키를 secret으로** — 같은 Credentials 페이지의 Secrets 탭에 이름 `wandb_api_key`, 값은 [wandb.ai/authorize](https://wandb.ai/authorize)의 키로 만든다. Pod를 띄울 때 Environment Variables에 `WANDB_API_KEY` = `{{ RUNPOD_SECRET_wandb_api_key }}`를 추가한다(열쇠 아이콘으로 선택 가능). 스크립트는 `WANDB_API_KEY`가 없으면 `RUNPOD_SECRET_wandb_api_key`와 `/etc/rp_environment`도 찾는다. 셋 다 실패하면 W&B만 건너뛰고 나머지는 진행한다.
4. **(직접) Pod 실행** — Pods → Deploy에서 다음과 같이 고르고 SSH로 접속해 실행한다.
   - GPU: RTX 4090, On-Demand. Community Cloud는 시간당 약 $0.34이고 개인 호스트라 품질 편차가 있다. Secure Cloud는 약 $0.69–0.74다.
   - 템플릿: Runpod PyTorch.
   - 디스크: 기본값.
   - 접속 정보는 Pod의 Connect 버튼에 나온다.
   ```bash
   curl -LsSf https://raw.githubusercontent.com/personal-waityet-dev/physical-ai-pratics/main/00-setup/cloud/rehearsal.sh | bash
   ```
5. Mac의 저장소 루트에서 결과를 회수한다. 스크립트가 끝날 때 명령을 출력한다. scp는 Connect 탭에 **"SSH over exposed TCP"**(공개 IP)가 있는 Pod에서만 된다(기본 SSH 프록시는 scp 미지원). 없으면 W&B run의 Files 탭에서 JSON을 받는다.
   ```bash
   scp -P <port> <user>@<host>:~/setup-results.tgz /tmp/ && tar xzf /tmp/setup-results.tgz -C 00-setup/
   ```
6. **Pod를 terminate(휴지통 아이콘)하고, 콘솔에서 Pod가 0개이고 Billing의 시간당 사용액이 $0인지 확인한다.** stop만 하면 디스크 비용이 계속 나간다.
7. 위 결과표의 "클라우드" 열을 채운다.

[`cloud/rehearsal.sh`](cloud/rehearsal.sh)가 처리하는 일은 다음과 같다. apt로 EGL/OSMesa·ffmpeg 설치, uv 설치, `00-setup`과 `common`만 sparse clone(PDF 90MB 제외), `uv sync --locked`, 점검 3종, 그리고 W&B 키가 있으면 업로드. MuJoCo는 EGL을 먼저 시도하고, 컨테이너에 NVIDIA EGL 라이브러리가 없으면 OSMesa(CPU 렌더)로 넘어간다. 이후 프로젝트도 `SPARSE_DIRS`와 실행 단계만 바꿔 같은 흐름을 쓴다.

## 남은 일

- [x] W&B 로그인 (2026-10-09, `~/.netrc`)
- [ ] 클라우드 리허설(위 절차) 후 결과표 클라우드 열 채우기
