# LoopLM Engineering: Master Root Cause Analysis (v1-v7)

이 문서는 `LoopLM` 개발 과정에서 발생한 모든 기술적 장애와 그 해결 과정을 유형별로 정리한 **지식 자산(Knowledge Asset)**입니다. GPU 커널 엔지니어링과 딥러닝 아키텍처 설계의 교차점에서 발생하는 복합적인 문제들을 다룹니다.

---

## 📂 유형 1: GPU 커널 인덱싱 및 하드웨어 정합성 (Hardware Law)

### **Problem [Indexing_Multiplier_Confusion]**
*   **증상**: `Thinking Trace` 출력 시 4개 토큰 단위로 데이터가 나오고 이후 12개가 0으로 나오는 "Skip-4" 패턴 반복.
*   **원인**: cuTile의 `ct.load` API에 대한 오해. `shape` 인자를 넘기면 인덱스는 자동으로 **타일 단위(Tile Index)**로 해석되는데, 여기에 수동으로 **엘리먼트 오프셋(`bid * 4`)**을 곱하여 실제 주소가 제곱으로 멀어짐.
*   **해결 (Fix)**: NVIDIA 공식 문서 확인 후 모든 커널 인덱싱을 `index=(bid, 0)`로 정규화.
*   **상태**: **[FIXED]** (v7)

### **Problem [Non_Power_of_Two_Crash]**
*   **증상**: `vocab_size=65`, `n_embd=384` 환경에서 커널 컴파일 에러 (`TileTypeError`).
*   **원인**: Blackwell 하드웨어와 cuTile 라이브러리는 성능을 위해 **2의 거듭제곱(Power of Two)** 타일 크기를 강제함.
*   **해결 (Fix)**: `next_pow2` 함수를 도입하여 텐서를 512, 128 크기로 패딩 후 커널에 전달하고 연산 후 다시 슬라이싱.
*   **상태**: **[FIXED]** (v3)

---

## 📂 유형 2: 수치적 안정성 및 확률 왜곡 (Numerical Integrity)

### **Problem [Zero_Exp_Bias]**
*   **증상**: 모델이 확신도가 낮음에도 불구하고 조기 종료하지 못하거나 무한 루프에 빠짐.
*   **원인**: `PaddingMode.ZERO` 사용 시 경계 밖이 0으로 채워짐. 소프트맥스 계산 중 `exp(0)=1`이 되어 실제 존재하지 않는 토큰들이 확률 분모(`sum_exp`)를 오염시킴.
*   **해결 (Fix)**: 커널 내부에서 `REAL_V` (실제 어휘 수)를 기반으로 **Column Masking**을 수행하여 패딩 영역의 `exp` 값을 0으로 강제함.
*   **상태**: **[FIXED]** (v7)

### **Problem [NaN_Halt_Trap]**
*   **증상**: 특정 조건에서 모든 사고 단계가 즉시 `0`으로 종료됨.
*   **원인**: 패딩 토큰 영역에서 `logits(-inf) - max_logit(-inf)` 연산 발생. IEEE 754 표준에 따라 `NaN`이 생성되고, 이것이 비교 연산(`NaN < Threshold`)에 들어가 논리 붕괴 유발.
*   **해결 (Fix)**: `is_real_token` 마스크와 `safe_max` 로직을 도입하여 `-inf` 영역에서의 연산을 안전하게 격리(NaN Guard).
*   **상태**: **[FIXED]** (v5)

---

## 📂 유형 3: 신경망 학습 역학 (Learning Dynamics)

### **Problem [Autograd_Graph_Disconnection]**
*   **증상**: 12번의 루프를 돌지만 1개 레이어 깊이의 모델보다 성능이 안 나옴 (학습 정체).
*   **원인**: `h = h_next`와 같은 인플레이스 할당을 사용하여 PyTorch의 Autograd 그래프가 끊어짐. 사실상 마지막 루프만 학습됨.
*   **해결 (Fix)**: `all_states` 리스트를 사용하는 **Trajectory Unrolling** 방식으로 변경하여 완전한 **BPTT(Backpropagation Through Time)** 구현.
*   **상태**: **[FIXED]** (v4)

### **Problem [Premature_Supervision_Noise]**
*   **증상**: 덧셈 학습 시 Loss가 떨어지지 않고 진동함.
*   **원인**: 루프의 아주 초기 단계(1~2단계)부터 정답을 맞히라고 강요함. 모델이 충분히 "생각"할 시간을 갖지 못한 채 노이즈 섞인 그래디언트만 수용함.
*   **해결 (Fix)**: **Half-Loop Supervision** (6~12단계에서만 Loss 계산)을 도입하여 모델의 자유로운 잠재 추론 공간 확보.
*   **상태**: **[FIXED]** (v6)

---

## 📂 유형 4: 전이 학습 및 데이터 정합성 (Transfer Learning)

### **Problem [Randomized_Head_Syndrome]**
*   **증상**: 셰익스피어 가중치를 이식했음에도 덧셈 학습이 0부터 시작하는 것보다 느림.
*   **원인**: `strict=False`로 가중치 로드 시, 어휘 사전 크기(`vocab_size`)가 다르면 `lm_head`가 로드되지 않고 랜덤 초기화 상태로 남음. 몸통은 천재인데 입(Head)이 무작위인 상태.
*   **해결 (Fix)**: `train_loop.py`에서 가중치 차원 불일치를 명시적으로 감지하고, 필요 시 Head만 리셋하거나 경고를 띄우는 로직 추가.
*   **상태**: **[FIXED]** (v5)

---

## 📂 유형 5: 아키텍처 설계 선택 (Architectural Design)

### **Observation [Persistent_X0_Injection]**
*   **내용**: 매 루프마다 `h = Block(h + x0)`을 수행하는 것이 타당한가?
*   **분석**: `x0`는 "입력 정보의 닻(Anchor)" 역할을 하여 장기 추론 시 입력 망각을 방지함. 하지만 상태값이 너무 커질 위험(Scaling issue)이 있음.
*   **결정**: 현재는 안정성을 위해 유지. 추후 **Phase 4: Ablation Study**를 통해 제거 시의 동역학 변화 측정 예정.
*   **상태**: **[MONITORED]** (v7)

---

### **💡 결론 및 엔지니어링 원칙**
1.  **Never Assume Implicit Scaling**: 하이레벨 라이브러리(cuTile)의 인덱싱 규칙은 공식 문서를 통해 차원별로 반드시 재검증해야 함.
2.  **Protect the Gradient Chain**: 루프 구조에서는 사소한 할당문 하나가 전체 지능 학습을 무력화할 수 있음.
3.  **Numerical Safety First**: GPU 커널에서는 `ZERO`가 단순한 0이 아니라 `exp(0)=1`과 같은 물리적 의미를 가짐을 명심해야 함.

---

## 🛠 파일별 문제 발생 및 해결 이력 (File-Specific History)

### **1. `model_loop.py` (Architecture & Integration)**
*   **[P1] Return Tuple Mismatch**: 초기 `forward`는 (logits, loss)만 반환했으나 Phase 2에서 `steps_taken`이 추가되며 `train_loop.py`에서 언팩 오류(ValueError) 발생. -> **[FIXED]**
*   **[P2] Autograd Breaking**: 인플레이스 할당(`h = h_next`)으로 인한 역전파 그래프 단절. 12단계 중 1단계만 학습됨. -> **[Trajectory List 도입으로 해결]**
*   **[P3] Inference-Evaluation Gap**: `eval()` 모드에서 Loss 계산을 건너뛰어 `estimate_loss` 시 AttributeError 발생. -> **[Inference 모드 Loss 계산 추가로 해결]**
*   **[P4] Memory Allocation Storm**: 루프 내부의 `F.pad` 호출로 인한 과도한 CUDA 할당 지연 및 주소 가변성 문제. -> **[Pinned State & In-place Reset 도입으로 해결]**
*   **[P5] Supervision Mismatch**: 모든 루프 단계에 정답 강요 시 초기 단계의 노이즈 그래디언트 문제. -> **[Half-Loop Supervision으로 해결]**

### **2. `looplm_kernels.py` (GPU Kernel Engineering)**
*   **[P1] Non-Power-of-Two Shapes**: 해결 완료.
*   **[P2] The "Skip-4" Multiplier Bug**: 해결 완료. (NVIDIA 공식 문서 기반 인덱싱 정규화)
*   **[P3] NaN-Halt Trap**: 해결 완료.
*   **[P4] ZERO-Exp Inflation**: 해결 완료. (Column Masking 도입)
*   **[P5] API Availability Mismatch**: 해결 완료.

---

## 📂 유형 6: 지능형 동역학 분석 (Intelligence Dynamics)

### **RCA v8: Algorithmic Overfitting vs. Generalization**
*   **Observation**: 셰익스피어 모델은 가변적 사고 깊이를 보여주나, 덧셈 모델은 모든 토큰에서 Max Loop(12)를 소모함. 또한 Val Loss가 상승함.
*   **Root Cause**: 
    1. **High Input Entropy**: 덧셈 데이터(`123+456=`)에서 `=` 이전의 토큰들은 모델 입장에서 다음에 어떤 숫자가 올지 예측하기 어렵습니다(엔트로피가 높음). 따라서 확신을 갖지 못하고 끝까지 생각하게 됩니다.
    2. **Memorization Shortcut**: 모델이 1개 레이어의 파라미터를 반복 사용하는 루프 구조임에도 불구하고, 5만 개의 샘플을 통째로 외워버리는 과적합이 발생함.
*   **Insight**: `LoopLM`의 지능을 제대로 확인하려면, **학습하지 않은 자릿수(Out-of-Distribution)**를 넣었을 때 루프 횟수를 늘려 정답을 맞히는 **"Extrapolation"** 테스트가 Phase 3의 핵심이 되어야 함.

---

---

## 📂 유형 7: Phase 3 Algorithmic Generalization 설계

### **Strategy [Systematic_Trace_Collection]**
*   **배경**: RTX 5070에서의 반복적인 수동 실행은 비효율적이며, 데이터 기반의 정밀한 분석이 필요함.
*   **해결**: `run_experiments.py` 통합 스크립트 도입. Training-Evaluation-OOD Testing을 자동화하고 모든 지표를 `summary.json`에 집계.
*   **상태**: **[IMPLEMENTED]** (v9)

### **RCA v10: Zero-shot Length Generalization Failure**
*   **Observation**: 
    *   `baseline`: Accuracy 5% (OOD), Avg Steps 11.05/12.
    *   `T1 (24 loops)`: Accuracy 0%, Avg Steps 24.0/24.
*   **Root Cause**: 
    1. **Insufficient Training (Pre-Grokking)**: 2000 iterations는 덧셈의 추상적 알고리즘을 학습하기에 부족함. 모델이 패턴은 익혔으나 자릿수 확장에 대응하는 '일반적 규칙'을 형성하지 못함.
    2. **Confidence-Accuracy Gap**: `baseline`은 루프를 줄였음에도 정답률이 낮음. 즉, "잘못된 답에 대해 확신(Overconfidence)"하는 현상 발생.
*   **Insight**: 
    *   단순히 루프를 늘리는 것(`T1`)만으로는 지능이 생기지 않음. 
    *   **Curriculum Learning** (2자리 -> 3자리 -> 4자리 순차 학습) 또는 **Longer Training** (10k+ iters)이 필수적임.
    *   Weight Decay와 Dropout의 조합이 일반화에 긍정적 영향을 미침 (`A3`가 `A1/A2`보다 나음).

---
