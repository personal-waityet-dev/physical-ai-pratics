# P01 `01-vlm-from-llm/` — LLM 엔지니어를 위한 VLM 해부

| 기간 | 환경 | 모듈 | 논문 |
|---|---|---|---|
| 2주 | Mac(MPS) | K1 | 보충 ViT, CLIP, SigLIP, LLaVA, PaliGemma |

## 왜 지금

2026년 physical AI의 모든 축(VLA, 체화 추론, WAM의 조건 입력, 옴니모달 백본)이 "비전 토큰을 언어 모델에 넣는" 구조 위에 서 있다. 멀티모달은 가장 큰 공백이므로 가장 먼저 메운다.

## 이론 목표

- 이미지가 토큰이 되는 과정: 패치화, 위치 임베딩, 해상도·패치 크기와 토큰 수의 관계(224px / 패치 14 → 256 토큰)
- 대조학습: CLIP(softmax InfoNCE) vs SigLIP(sigmoid loss, 배치 크기 의존성 차이)
- 의미 특징(SigLIP) vs 공간·기하 특징(DINOv2) — Prismatic/OpenVLA/CogACT가 둘을 합치는 이유
- LLaVA식 projector와 2단계 학습, PaliGemma의 prefix-LM(이미지+프롬프트는 양방향, 출력은 causal)

## 실행 (최소 범위)

1. 패치화를 텐서 reshape로 직접 구현해 HF SigLIP의 patch embedding 출력과 일치 확인. 해상도 224/384/512에서 토큰 수·메모리·지연 측정
2. SigLIP으로 zero-shot 분류와 이미지-텍스트 검색(recall@k). 인코더를 고정하고 작은 projection head를 softmax loss와 sigmoid loss로 각각 학습해 비교
3. DINOv2와 SigLIP의 패치 특징을 PCA로 색칠해 비교(물체 경계·배치가 어디에 더 잘 보이는가)
4. **tiny VLM**: 고정 SigLIP + 2층 MLP projector + 고정 SmolLM2-360M. Flickr8k 또는 COCO 캡션 일부로 projector만 학습(LLaVA stage 1)
5. 같은 tiny VLM을 causal mask와 prefix-LM mask로 각각 학습해 비교하고, mask 행렬을 시각화

## 확장

- LM에 LoRA 추가 학습(stage 2)
- SmolVLM2 코드 읽기(이미지 토큰 삽입 위치, pixel shuffle로 토큰 압축)
- ResNet에 FiLM 층을 넣어 텍스트 조건부 특징 만들기(P05 예습)

## 산출물

- `tiny_vlm/` 코드
- 해상도별 토큰 수·지연 표
- 특징 PCA 그림
- mask 비교표
- `docs/notes/vlm-basics.md`

## 완료 기준

1. 토큰 수를 손으로 계산해 실측과 일치
2. tiny VLM이 처음 보는 이미지에서 물체·색 수준의 캡션을 생성하고, projector 학습 전후 캡션 loss 감소를 수치로 제시
3. prefix-LM과 causal의 차이를 자기 코드로 설명

## 넘기는 것

- tiny VLM(P02 이미지 과제, P06 백본 후보)
- mask 코드(P09 blockwise mask)
