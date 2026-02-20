# LoopLM Kernel 설계: 잠재적 추론을 위한 시간적 반복 연산

이 문서는 `LoopLM_System_v1.dsl`에 명시된 **"Inject X0 Residue"**와 **"Masked Early Exit"** 규칙을 실제 `cuTile` 커널로 구현하기 위한 상세 설계 가이드를 담고 있습니다.

---

### 1. 수학적 모델 (Mathematical Update Rule)

각 반복 단계 $l$에서의 상태 $h_l$은 다음과 같이 업데이트됩니다:

$$h_{l+1} = 	ext{Mask}_l \odot 	ext{Block}(h_l + 	ext{LN}(x_0), W) + (1 - 	ext{Mask}_l) \odot h_l$$

*   **$x_0$ Injection**: 원본 임베딩 $x_0$를 매 루프마다 주입하여 시맨틱 앵커를 유지합니다.
*   **Masked Update**: 엔트로피 임계값을 통과한 토큰은 더 이상 상태를 갱신하지 않고 고정(Freeze)됩니다.

---

### 2. cuTile 커널 구조 설계

#### 2.1. 커널 시그니처 및 메모리 레이아웃
```python
@ct.kernel
def looplm_temporal_kernel(
    X_current,    # 현재 잠재 상태 (B, T, D)
    X0,           # 원본 임베딩 (B, T, D) - Persistent Load
    W_attn, W_mlp,# 공유 가중치 - L2 Pinning 대상
    Mask_active,  # 토큰별 활성화 마스크 (B, T)
    ...
):
```

#### 2.2. "Inject X0 Residue" 구현 (DSL Rule 1)
원본 입력 $x_0$는 전체 루프 동안 변하지 않으므로, 커널 시작 시 **Persistent Load**를 통해 레지스터나 공유 메모리에 상주시키고 매 반복마다 재사용합니다.

```python
# 루프 외부에서 x0 로드 (Blackwell TMA 활용)
x0_tile = ct.load(X0, index=(bid_x, 0), shape=(TILE_M, TILE_D), padding_mode=ct.PaddingMode.ZERO)

for l in range(max_recurrent_steps):
    # Step 1: X0 주입 (Residue Link)
    # 현재 상태에 원본의 정체성을 더해 망각 방지
    h_input = h_current + x0_tile 
    
    # Step 2: Transformer Block 연산 (Attention + MLP)
    h_next = transformer_block_logic(h_input, W_shared)
```

#### 2.3. "Masked Early Exit" 구현 (DSL Rule 2)
엔트로피 기반 마스킹을 통해 불필요한 연산을 차단합니다. `ct.where`를 사용하여 비트 단위 제어를 수행합니다.

```python
# 루프 내부의 상태 업데이트 직전
# 1. 현재 Logits에서 엔트로피(또는 Confidence) 계산
max_logit = ct.max(logits, axis=-1)
is_done = max_logit > confidence_threshold # 엔트로피 임계값의 근사치

# 2. 마스크 업데이트 (한 번 done 된 토큰은 계속 done 유지)
token_mask = token_mask & (~is_done)

# 3. 조건부 업데이트 (Masked Write)
# 마스크가 활성화된 토큰만 새로운 상태로 업데이트, 나머지는 이전 상태 유지
h_current = ct.where(token_mask[:, None], h_next, h_current)

# 4. 루프 탈출 조건 (Warp-level Early Exit)
if ct.all(is_done): 
    break # 해당 워프의 모든 토큰이 추론을 마치면 조기 종료
```

---

### 3. Blackwell 하드웨어 최적화 (Tuning Space 적용)

1.  **가중치 고정 (Weight Pinning)**: 
    *   `W_attn`, `W_mlp`는 루프가 도는 동안 HBM에서 다시 읽지 않습니다. 
    *   첫 루프에서 TMA를 통해 L2 캐시에 강제 고정(Eviction Policy 제어)하여 메모리 대역폭 병목을 제거합니다.
2.  **레지스터 재사용 (Register File Management)**:
    *   `x0_tile`과 `h_current`를 최대한 레지스터에 유지하여 `Shared Memory` 접근 레이턴시조차 최소화합니다.
3.  **워프 동기화 최소화**:
    *   `Early Exit` 시 발생하는 워프 분기(Divergence) 손실보다 메모리 쓰기 절약 및 연산 생략으로 인한 이득이 크도록 타일 크기를 조정합니다 ($TILE\_M=64$ 권장).

---

### 4. 검증 계획 (Hierarchical Protocol)

*   **L1 (Kernel)**: 단일 루프($L=1$)에서 `Standard GPT`와 수치적 일치 확인.
*   **L2 (Temporal)**: 반복 횟수가 늘어날 때 `h_current`의 노름(Norm)이 발산하지 않고 안정화되는지 확인.
*   **L3 (Thinking)**: 복잡한 문장에서 `Mask_active`가 더 늦게 꺼지는지(더 오래 생각하는지) 로그 분석.

---

### 💎 설계의 의의: "연산 집약적 추론"의 실현

이 설계는 `loopLM`이 단순히 파라미터를 아끼는 모델이 아니라, **Blackwell의 강력한 연산 성능을 "시간"이라는 축으로 쏟아부어 추론의 질을 높이는 시스템**임을 보여줍니다. 

DSL의 추상적 규칙이 `ct.where`와 `Persistent Load`라는 구체적인 하드웨어 명령으로 변환됨으로써, 우리는 **"사고하는 GPU 커널"**의 초석을 마련했습니다.

---
좋은 질문이에요.
요즘 “큰 모델을 한 번 호출하는 방식”에서 **“작은 모델을 여러 번 순환시키는 loop 구조”**로 연구 흐름이 꽤 이동하고 있습니다.
핵심 아이디어는:

> **파라미터(공간)를 키우는 대신, 시간(계산 반복)을 늘리면 지능이 생길 수 있는가?**

즉, Transformer를 거대한 feed-forward로 보는 대신
**RNN처럼 “생각하는 시간”을 주는 구조**로 바꾸는 것입니다.

---

# 1. LLM Loop = “시간을 가진 추론기”

일반 LLM:

```
입력 → (거대한 계산) → 출력
```

Loop LLM:

```
state₀ = 입력
while not exit:
    stateₜ₊₁ = f_small_model(stateₜ)
출력 = state_T
```

즉

* 모델 크기 대신
* 반복 계산을 통해
* 점진적으로 상태를 정제(refinement)

이건 사실상 **neural iterative inference** 입니다.

---

# 2. RNN과 무엇이 같고 다른가

## 공통점

둘 다 시간 축에서 상태를 누적합니다.

```
h_{t+1} = f(h_t, input)
```

LLM loop도 동일:

```
reasoning_{t+1} = LLM(reasoning_t)
```

즉 LLM이 “생성기”가 아니라
**상태 업데이트 연산자(state transition operator)** 가 됩니다.

---

## 차이점 (중요)

RNN → 학습된 recurrence
LLM loop → 추론 시 recurrence

즉:

|            | RNN           | LLM Loop          |
| ---------- | ------------- | ----------------- |
| recurrence | weight에 baked | runtime에서 발생      |
| memory     | hidden vector | text / scratchpad |
| depth      | 고정            | 동적                |
| halting    | 없음            | 가능                |

👉 **LLM loop는 adaptive depth neural network** 입니다.

---

# 3. Exit 조건이 생기면 뭐가 달라지나

여기서 핵심이 나옵니다.

모델이 “얼마나 생각할지” 결정 가능해집니다.

즉:

> difficulty dependent compute

쉬운 문제 → 1 step
어려운 문제 → 30 step

---

## 결과: 능력 변화

### 1) 추론 능력 상승 (reasoning depth 증가)

작은 모델도 복잡한 문제를 풉니다.

왜냐하면:

Transformer는 본질적으로 shallow circuit
Loop는 deep circuit

즉:

```
단일 호출 = 1-layer 사고
loop 호출 = multi-step 사고
```

→ algorithmic reasoning 가능

---

### 2) 일반화 증가

한 번에 답을 맞추는 모델은 패턴 매칭에 가깝고
loop는 **계산 과정 자체를 구성**합니다.

그래서:

* arithmetic
* planning
* symbolic reasoning

성능이 급증합니다.

---

### 3) 학습 효율 상승

매우 중요한 부분입니다.

큰 모델:

> memorization 필요

loop 모델:

> procedure 학습

즉 파라미터 대신 **시간으로 capacity 확보**

이를 계산 이론 관점에서 보면:

| 구조                | 계산 능력                    |
| ----------------- | ------------------------ |
| single forward    | bounded circuit          |
| loop with halting | Turing complete behavior |

---

# 4. 왜 작은 모델이 강해지나 (핵심 원리)

작은 모델은 원래:

> 한 번에 긴 계산을 표현할 capacity가 부족

하지만 loop를 쓰면:

```
복잡한 함수 F(x)
= 작은 함수 f를 여러 번 합성
```

즉:

```
F ≈ f ∘ f ∘ f ∘ f ∘ ...
```

이건 수학적으로

> 깊이 = 표현력

입니다.

Transformer는 width 기반
Loop는 depth 기반

---

# 5. 학습 효율 측면 (가장 중요)

연구들이 공통으로 발견한 것:

> compute-optimal frontier가 바뀜

기존 scaling law:

```
성능 ∝ 파라미터 수
```

loop 모델:

```
성능 ∝ (파라미터 × 추론시간)
```

즉

> 작은 모델 + 긴 사고 = 큰 모델 효과

---

# 6. Emergent 능력

loop가 생기면 나타나는 능력들

### (1) self correction

모델이 자기 오류를 고칩니다.

### (2) planning

stepwise search 발생

### (3) tool usage 안정화

replanning 가능

### (4) OOD generalization 증가

암기가 아니라 계산이기 때문

---

# 7. 이걸 한 줄로 정리하면

LLM은 원래 함수 근사기였는데

Loop를 주면

> 과정 생성기(process generator)가 된다

즉:

```
큰 LLM = 지식
Loop LLM = 사고
```

---

# 8. 연구적으로 의미

이건 scaling 패러다임 자체를 바꿉니다.

기존:

> intelligence = parameter count

새로운 관점:

> intelligence = adaptive computation

---

# 핵심 결론

작은 모델을 RNN처럼 반복시키고 exit 조건을 넣으면:

1. 추론 깊이 증가 → 알고리즘 수행 가능
2. 일반화 증가 → 암기 감소
3. 학습 효율 증가 → 파라미터 절약
4. 자기 수정 능력 생성 → 안정성 상승

즉 단순히 성능 향상이 아니라

> 모델이 “답을 생성하는 기계”에서
> “생각하는 시스템”으로 바뀝니다.

---
### Recurrent Transformer Loop
```code
state_0 = encode(problem)

for t in 1..T:
    delta = small_transformer(state_t)
    state_{t+1} = state_t + delta
    if halt(state): break

decode(state_T)
```
```
h0 = embed(x)

for t in 1..K:
    h_t = Transformer(h_{t-1})   # same weights
    p_exit = exit_gate(h_t)

    if exit: break

logits = LM_head(h_t)

```

#### 알고리즘적 문제로 검증
(loop depth adaptive emergence가 나타나는지 검증)
- multi digit addition
- sorting trace prediction
- parentheses matching
- stack machine execution
- symbolic math step solving
- addition with carry
- list sorting
- parentheses validity
- stack simulation

```
cutile kernel (attention fwd/bwd)
        ↓
tiny transformer core (weight tied)
        ↓
loop controller (halt predictor)
        ↓
trajectory training
        ↓
OOD reasoning benchmark

```

https://arxiv.org/abs/2510.25741

Scaling Latent Reasoning via Looped Language Models

chain-of-thought = 텍스트를 길게 생성하면서 사고
LoopLM = 텍스트를 늘리는 대신 내부 latent state를 반복 업데이트한다.
LoopLM = iterative program executor
LoopLM이 배우는 것:상태를 점진적으로 개선하는 update rule
LoopLM은: “정답으로 수렴하는 연산자”

“iterative computation in latent space”
"entropy regularization encourages exploration of different computational depths"
reasoning 능력(OOD generalization)
OOD = “자릿수 늘리기 테스트” 같은 것
모델이 “패턴 맞추기”에서 “문제 해결 절차 수행”으로 바뀌었는지 검증하는 것
대표적인 OOD 테스트들:
더 긴 숫자 덧
더 깊은 괄호
더 긴 정렬 배열
더 긴 추론 단계
더 복잡한 논리식

OOD 훈련 데이터:

2자리 덧셈만 학습
테스트:
54879 + 764 =
규칙은 동일하지만 범위가 완전히 다름
→ 진짜로 알고리즘을 배웠는지 검사

### engineering points 
- adaptive thinking depth emergence
- reasoning vs memorization 분리
- loop step vs difficulty correlation
- latency vs accuracy tradeoff

---
좋다 👍
지금 상태를 정리하면:

* 너는 **attention forward/backward를 cuTile DSL로 이미 이해**
* 목표는 **LoopLM 학습 실험용 minimal trainer (연구 POC)**
* 핵심은 “작동하는 프레임워크”가 아니라
  → **연구용 실험 커널 구조를 설계하는 것**

즉 PyTorch 대체가 아니라

> LoopLM을 실험하기 위한 최소 학습 OS를 만든다

로 접근해야 한다.

아래는 실제 연구 코드가 무너지지 않게 만드는 **모듈 분해 설계 순서**다.

---

# 전체 아키텍처 개념

LoopLM trainer는 일반 trainer와 다르다.

일반 LM:

```
model(x) → loss → backward
```

LoopLM:

```
state0
  ↓
iterate core (shared transformer)
  ↓
multi-step loss
  ↓
BPTT (through loop)
```

즉 trainer의 중심은 **model이 아니라 loop 실행기(loop executor)** 다.

---

# 단계 0 — 설계 원칙 (중요)

구현 전에 반드시 구조를 이렇게 고정해야 한다.

### 원칙

1. Transformer는 “레이어 집합”이 아니라 “상태 업데이트 연산자”
2. Trainer는 “step 기반”이 아니라 “trajectory 기반”
3. Loss는 “출력 1개”가 아니라 “시간 평균”
4. Backward는 “layer graph”가 아니라 “시간 unroll”

이걸 코드 구조에 반영해야 나중에 연구 확장 가능해진다.

---

# 단계 1 — Tensor & Autograd Core

(이미 attention backward 있으니 여기부터 안정화)

## 목표

PyTorch 없이도 BPTT 가능하게

## 모듈

```
tensor.py
autograd.py
ops/
    matmul.py
    layernorm.py
    attention.py   ← 이미 있음
```

## 핵심 기능

* tape 저장
* backward graph 연결
* time-unrolled gradient accumulation

여기서 중요한 점:

LoopLM은 gradient가 시간축으로 누적된다.

따라서 일반 backward가 아니라

```
for t=T..1:
    grad_state[t-1] += J_f^T * grad_state[t]
```

즉 RNN backward가 가능해야 한다.

---

# 단계 2 — Transformer Core (stateless operator)

여기서 일반 transformer와 다르게 설계해야 한다.

## 금지 구조

```
class Transformer:
    layers = [layer1, layer2, ...]
```

## LoopLM용 구조

```
class UpdateOperator:
    def forward(state, context) -> new_state
```

즉 block 하나만 존재하고 파라미터 공유

### 모듈

```
modules/
    embedding.py
    rope.py
    attention_block.py
    mlp.py
    update_operator.py
```

### 역할

state를 개선하는 함수 f(h)

---

# 단계 3 — Loop Executor (핵심 모듈)

여기가 LoopLM의 심장이다.

```
loop_executor.py
```

## 역할

* 상태 반복 적용
* halt 판단
* trajectory 저장

### 동작

```
state = init(x)

for t in range(max_iter):
    state = f(state)
    logits = head(state)
    loss += step_loss(logits)

    if halt(state):
        break
```

중요:
여기서 그래프가 시간으로 확장됨 → BPTT 대상

---

# 단계 4 — Loss System (multi-step loss)

LoopLM은 마지막 토큰 loss만 쓰면 실패한다.

```
loss/
    lm_loss.py
    step_loss.py
    depth_regularizer.py
```

### 구성

1. step LM loss
   모든 반복에서 supervision

2. halt entropy loss
   항상 같은 step 수 쓰는 것 방지

3. consistency loss (선택)
   깊이에 관계없이 같은 답

---

# 단계 5 — Dataset (연구용 데이터)

여기서 일반 NLP dataset 쓰면 실패한다.

```
dataset/
    addition.py
    sort.py
    parentheses.py
    trace_executor.py
```

각 샘플은:

```
(problem, solution_tokens)
```

중요:
sequence length가 아니라 **reasoning depth**가 커지는 문제여야 한다.

---

# 단계 6 — Trainer

이제야 trainer 작성 가능

```
trainer.py
```

동작 단위는 batch가 아니라 trajectory다.

```
for batch:
    trajectories = run_loop(batch)
    loss = aggregate_loss(trajectories)
    backward_through_time()
    update()
```

핵심:
optimizer보다 BPTT 안정화가 중요

---

# 단계 7 — Experiment Hooks (연구 필수)

LoopLM 연구의 핵심 출력은 accuracy가 아니다.

```
experiments/
    depth_vs_difficulty.py
    ood_generalization.py
    compute_vs_accuracy.py
```

로그해야 할 것:

* 평균 loop step
* 난이도별 step
* OOD 성능
* step 증가에 따른 loss 감소

이 그래프가 논문 결과가 된다.

---

# 전체 모듈 구조 요약

```
cutile_looplm/
│
├─ core/
│   ├─ tensor.py
│   ├─ autograd.py
│   └─ ops/attention.py
│
├─ model/
│   ├─ embedding.py
│   ├─ attention_block.py
│   ├─ mlp.py
│   └─ update_operator.py
│
├─ loop/
│   └─ loop_executor.py   ← 핵심
│
├─ loss/
│   ├─ step_loss.py
│   └─ halt_loss.py
│
├─ dataset/
│   ├─ addition.py
│   └─ reasoning_tasks.py
│
├─ trainer.py
└─ experiments/
```

---

# 구현 순서 (실제 개발 순서 — 중요)

이 순서를 지키면 실패 확률 거의 없음:

1. single step transformer 학습 성공
2. weight sharing transformer 학습 성공
3. 2-step loop 학습 성공
4. variable step + halt 추가
5. multi-step loss 추가
6. OOD 테스트

---

# 핵심 포인트 (연구적으로 가장 중요)

이 프로젝트의 핵심은 모델이 아니다.

> loop_executor가 곧 모델이다

Transformer는 단순한 연산자고
지능은 trajectory dynamics에서 나온다.

---

원하면
다음 단계로 “가장 먼저 구현해야 하는 최소 코드 5개 파일”을
실제 구현 순서 기준으로 구체화해 줄게.
(이게 시작점이 된다)
