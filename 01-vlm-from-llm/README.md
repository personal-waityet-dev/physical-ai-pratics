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

## 실행 기록

### 실행 1 — 패치화

#### 예측

**Q1. 224px, 패치 14 → 토큰 몇 개?**

- 16 * 16 = 256

**Q2. 384px는? 384는 14로 나누어떨어지지 않는다 — 남는 픽셀은 어떻게 되나?**

- 28*28 = 784 , 남는 픽셀을 마지막 패치에 담고 나머지는 빈값?으로 채우도록 해본다.

**Q3. 224 모델에 512px 이미지를 넣으면 모델의 어느 부분이 문제가 되나? LLM에서 학습 길이보다 긴 context를 넣을 때와 비교해 본다.**

- 입력 토큰이 길어지게 된다. → 위치임베딩(학습된것)이 맞지 않게 된다.
- 512 이미지면 패치수가 37 * 37 (Q2에서 답한 패딩방식인 경우) = 1369 가 된다. 위치표는 256 이므로, 학습된 위치표를 1369 길이에 맞게 수정해 주어야한다.
- LLM 의 경우는 학습된 입력보다 긴 컨텍스트가 들어오는 경우, 위치표(GPT-2) 에서 벗어나더라도, 1차원 거리이므로, 거리 보간을 수행하면 되지만(상대 위치가 처리되는 모델은 별도 조정 필요없을듯? ), 이미지 패치는 2차원 정보가 포함되는 것을 고려하면, 16 * 16 -> 37 * 37 인 것을 고려한 2차원 보간이 필요하다
- (Claude) 보충: 상대 위치(RoPE) 모델도 학습 길이를 넘기면 그대로는 안 된다. 위치 표가 없어 계산은 돌아가지만, 학습 때 본 적 없는 거리(회전 각도)가 나와 성능이 무너진다. 그래서 LLM의 문맥 확장도 위치를 학습 범위 안으로 눌러 넣는 위치 보간([2306.15595](https://arxiv.org/abs/2306.15595))이나 YaRN([2309.00071](https://arxiv.org/abs/2309.00071))으로 조정한다. ViT가 고해상도에서 위치 표를 2차원 보간하는 것(ViT 논문 §3.2)과 같은 생각이다.

#### 결과

**Q1**

- 256

**Q2**

- 729 (27*27) . 모델 구현에서는 패딩으로 채워서 올림하는 것이 아니라, 버림
- (Claude) 근거: 출력의 `padding=valid`(덧대지 않음, 창이 이미지 안에 온전히 들어가는 위치만), 384 모델의 `Embedding(729, 1152)`. 출력 = ⌊(384 + 2·0 − 14)/14⌋ + 1 = 27이고, 27×14 = 378이라 오른쪽·아래 6픽셀이 버려진다. PyTorch `Conv2d`는 패딩 양을 사람이 정하고(정수, `"valid"`, 이동 간격 1일 때만 `"same"`) 입력에 맞춰 자동으로 덧대지 않는다. 예측한 28×28은 TensorFlow `padding="SAME"`처럼 ⌈384/14⌉ = 28이 되게 자동으로 덧대는 방식의 결과다.

**Q3**

- 512px 이미지를 그대로 넣으면 다음 에러가 난다.
  ```text
  The size of tensor a (1296) must match the size of tensor b (256) at non-singleton dimension 1
  ```
  - position table (256) 이 입력 토큰수와 맞지 않음
- 현재 사용 모델의 forward option 중 `interpolate_pos_encoding=True` 를 사용하면, 보정 구현이 들어가 있음

**패치 임베딩 비교와 허용 오차**

- (Claude) float64로 두 구현을 비교하면 차이 1.2e-14 → 알고리즘이 같고, float32 차이는 반올림 오차뿐이다.
- (Claude) 출력 하나는 K = 3×14×14 = 588항 내적, float32 단위 반올림 u = 2⁻²⁴ ≈ 6e-8. 반올림 오차의 보통 크기 √K·u·Σ|w·x| ≈ 1.7e-5, 최악 상한 K·u·Σ|w·x| ≈ 4.2e-4 (Σ|w·x| 최대 12.06).
- (Claude) 실측 최대 오차 5.7e-6(CPU 5.5e-6과 같은 수준)으로 보통 크기 이하 → 허용 오차는 atol ≈ 2e-5 ~ 5e-5 하나로 두면 근거가 선다. NVIDIA GPU는 합성곱에 TF32(가수 10비트)를 기본으로 써서 상대 오차 1e-3 수준이 되므로 완화가 필요하지만, MPS float32는 CPU와 같은 수준이었다.

**실행1 최종(포지션 인코딩 보간옵션준것)의 전체 출력**

```text
device: mps | image: (640, 480)
patch_embedding: Conv2d(3, 1152, kernel_size=(14, 14), stride=(14, 14), padding=valid)
position_embedding: Embedding(256, 1152)
{'shape_mine': torch.Size([1, 256, 1152]), 'shape_hf': torch.Size([1, 256, 1152]), 'max_abs_err': tensor(5.7220e-06, device='mps:0', grad_fn=<MaxBackward1>), 'mean_abs_err': tensor(1.5992e-07, device='mps:0', grad_fn=<MeanBackward0>), 'atol': 1e-05, 'rtol': 0.0001, 'allclose': True}
384 embedding
patch_embedding: Conv2d(3, 1152, kernel_size=(14, 14), stride=(14, 14), padding=valid)
position_embedding: Embedding(729, 1152)
model | device | dtype | warmup | repeat | error | hidden_state_shape | timed | memory
google/siglip-so400m-patch14-224 | cpu | torch.float32 | 2 | 5 |  | torch.Size([1, 256, 1152]) | 267.8910419344902 | ({'hidden_state_shape': torch.Size([1, 256, 1152])}, nan)
google/siglip-so400m-patch14-224 | mps | torch.float32 | 2 | 5 |  | torch.Size([1, 256, 1152]) | 65.95487473532557 | ({'hidden_state_shape': torch.Size([1, 256, 1152])}, 13.0)
google/siglip-so400m-patch14-384 | cpu | torch.float32 | 2 | 5 |  | torch.Size([1, 729, 1152]) | 721.4490831829607 | ({'hidden_state_shape': torch.Size([1, 729, 1152])}, nan)
google/siglip-so400m-patch14-384 | mps | torch.float32 | 2 | 5 |  | torch.Size([1, 729, 1152]) | 178.2537503167987 | ({'hidden_state_shape': torch.Size([1, 729, 1152])}, 37.0)
google/siglip-so400m-patch14-224 | cpu | torch.float32 | 2 | 5 |  | torch.Size([1, 1296, 1152]) | 1297.731916885823 | ({'hidden_state_shape': torch.Size([1, 1296, 1152])}, nan)
google/siglip-so400m-patch14-224 | mps | torch.float32 | 2 | 5 |  | torch.Size([1, 1296, 1152]) | 329.6655840240419 | ({'hidden_state_shape': torch.Size([1, 1296, 1152])}, 66.0)
```

### 실행 2 — zero-shot·검색, softmax vs sigmoid

#### 예측

**Q1. SigLIP으로 zero-shot 분류를 할 때 클래스별 확률을 모두 더하면 1이 되나? 안 된다면 분류 결과를 어떻게 읽어야 하나?**

- 클래스별 확률의 합은 1이 되지 않는다. 멀티라벨에 대한 바이너리 분류로 읽어야한다.

**Q2. 인코더를 얼리고 작은 projection head를 학습할 때, 배치를 줄이면 softmax 손실과 sigmoid 손실 중 어느 쪽 성능이 더 많이 떨어질까?**

- 소프트맥스 손실쪽이 많이 떨어진다. (손실 계산에서 배치 전체의 라벨 분포를 계산하는데, positive-negative 비율이 극단적으로 차이나는 실제 데이터 속성상 배치가 작을 수롤 실제 정답 분포에서 멀어지기 때문이다.)

**Q3. 보통의 텍스트 인코더(BERT, CLIP)는 어텐션 마스크와 정해진 토큰 위치(CLS, EOS)에서 임베딩을 꺼내서 패딩 길이와 무관하다. SigLIP은 시퀀스 맨 마지막 칸에서 임베딩을 꺼내고, 학습 때 항상 고정 길이(이 체크포인트는 16)로 패딩하고 가리지 않았다. 같은 문장(예: `'a photo of a cat'`)의 zero-shot 점수가 ① 처리기 기본값(패딩 없음) ② `padding="max_length"` ③ 길이가 다른 문장과 한 배치로 `padding="longest"`에서 각각 얼마나 달라질까?**

- 기본구현에서, 학습시 텍스트 패딩은 max length 로 구현되었다. 

#### 결과

**Q1**

- 

**Q2**

- 

**Q3**

- 

**전체 출력**

```text
```

### 실행 3 — SigLIP vs DINOv2 패치 특징 PCA

#### 예측

**Q1. 어느 모델의 패치 특징 PCA에서 물체 경계·부분이 더 또렷할까? 학습 목적으로 설명해 본다.**

- 

**Q2. SigLIP 비전 모델에는 분류 토큰(CLS)이 없다. 그럼 이미지 전체 표현은 어디서 나오고, 패치 특징은 어디서 꺼내나?**

- 

**Q3. DINOv2(`facebook/dinov2-small`)의 출력 토큰 수는 패치 수보다 많다. 왜 그런가?**

- 

#### 결과

**Q1**

- 

**Q2**

- 

**Q3**

- 

**전체 출력**

```text
```

### 실행 4 — tiny VLM (projector만 학습)

#### 예측

**Q1. 두 모델 config에서 hidden size를 찾아, 2층 MLP projector의 학습 파라미터 수를 계산한다. 전체 파라미터 중 몇 %인가?**

- 

**Q2. 이미지 1장 = 토큰 몇 개, 캡션 1개 = 토큰 몇 개인가? 이 비율에서 loss는 어느 토큰에만 걸어야 하나? LLM SFT의 무엇과 같은가?**

- 

**Q3. 학습 전(무작위 projector) 캡션 loss는 대략 어느 수준일까? 이미지 없이 캡션만 넣은 loss와 비교하면 어떤 관계여야 하나?**

- 

#### 결과

**Q1**

- 

**Q2**

- 

**Q3**

- 

**전체 출력**

```text
```

### 실행 5 — causal vs prefix-LM mask

#### 예측

**Q1. SmolLM2는 causal로 사전학습됐다. 여기에 prefix-LM mask를 씌우면 무엇이 어긋나나? PaliGemma는 이 문제를 어떻게 피했나?**

- 

**Q2. 공정한 비교를 위해 무엇을 고정해야 하나?**

- 

#### 결과

**Q1**

- 

**Q2**

- 

**전체 출력**

```text
```
