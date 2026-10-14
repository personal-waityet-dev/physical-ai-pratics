# vlm basics

## 이론 목표 및 이해

### 1. 이미지가 토큰이 되는 과정

> 패치화, 위치 임베딩, 해상도·패치 크기와 토큰 수의 관계(224px / 패치 14 → 256 토큰)

**ViT** ([2010.11929](https://arxiv.org/abs/2010.11929)), §3.1만

- 패치 임베딩이 "픽셀 패치의 선형 사영"이라는 것은 어떤 `Conv2d` 하나와 같은 연산인가? LLM의 토큰 임베딩(lookup)과는 무엇이 다른가?
- 위치 임베딩은 GPT-2의 무엇과 같고, RoPE 계열과는 무엇이 다른가?

### 2. 대조학습

> CLIP(softmax InfoNCE) vs SigLIP(sigmoid loss, 배치 크기 의존성 차이)

**CLIP** ([2103.00020](https://arxiv.org/abs/2103.00020)) → **SigLIP** ([2303.15343](https://arxiv.org/abs/2303.15343))

- softmax(InfoNCE) 손실은 배치 안에서 무엇을 보고, 왜 배치 크기에 민감해지나? 임베딩 모델 학습에서 쓰던 in-batch negatives와 연결해 본다.
- sigmoid 손실은 무엇을 바꾸나? 배치를 여러 장치에 나눌 때 무엇이 쉬워지나?
- SigLIP은 학습 가능한 온도 t와 편향 b를 두고, b를 **−10**으로 초기화한다. 배치 안의 positive:negative 쌍 비율을 생각하면 왜 그렇게 큰 음수에서 시작하나?

### 3. 의미 특징(SigLIP) vs 공간·기하 특징(DINOv2)

> Prismatic/OpenVLA/CogACT가 둘을 합치는 이유

**DINOv2** ([2304.07193](https://arxiv.org/abs/2304.07193)) — 실행 3 직전에

- 이미지-텍스트 대조학습(SigLIP)과 자기지도 학습(DINOv2)은 각각 패치 특징에 무엇을 남기도록 압력을 주나?

### 4. LLaVA식 projector와 2단계 학습, PaliGemma의 prefix-LM

> 이미지+프롬프트는 양방향, 출력은 causal

**LLaVA** ([2304.08485](https://arxiv.org/abs/2304.08485)) → **PaliGemma** ([2407.07726](https://arxiv.org/abs/2407.07726))

- projector를 거친 이미지 토큰은 LM 입장에서 어떤 LLM 미세조정 기법과 가장 닮았나? (입력 앞에 학습된 벡터를 붙이는 계열)
- LLaVA 2단계에서 각 단계에 무엇을 얼리고 무엇을 학습하나? 왜 그 순서인가?
- PaliGemma의 prefix-LM mask에서 어떤 토큰들끼리 양방향으로 보나? T5·UniLM에서 본 것과 무엇이 같은가?
