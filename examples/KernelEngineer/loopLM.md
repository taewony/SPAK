# loopLM: 잠재적 추론(Latent Reasoning)을 위한 반복형 언어 모델 설계 및 검증 전략

**loopLM**은 고정된 깊이(Depth)의 트랜스포머를 넘어, 동일한 디코더 블록을 여러 번 반복(Loop)하여 계산 깊이를 "시간" 차원으로 확장함으로써 **잠재적 추론 능력**을 확보하는 차세대 아키텍처입니다.

---

### 1. 설계 및 검증 로드맵 (Hierarchical Roadmap)

참조 구현체가 없는 새로운 아키텍처이므로, 검증된 **Standard GPT(12L)**를 기준으로 단계별 등가성과 추론 이득을 증명합니다.

#### **Step 0: Standard Baseline Sanity (The Gold Standard)**
*   **목표**: 12개 레이어를 쌓은 표준 트랜스포머(`GPT-12L`)의 무결성 확보.
*   **실행**: `nanoGPT/train.py`를 통해 셰익스피어 데이터셋에서 `Val Loss < 1.5` 달성.
*   **의의**: 모든 `loopLM` 실험의 수치적, 성능적 기준점(Anchor)이 됨.

#### **Step 1: Space-Time 등가성 테스트 (The Architectural Bridge)**
*   **목표**: `Standard GPT(12L)` vs `loopLM(1L x 12it)` 비교.
*   **검증**: 파라미터는 1/12이지만 연산량(FLOPs)이 동일한 두 모델의 손실값 수렴 곡선을 대조.
*   **핵심 지표**: 동일한 FLOPs 환경에서 `loopLM`이 `GPT-1L`보다는 우수하고 `GPT-12L`에 얼마나 근접하는지 측정.

#### **Step 2: 잠재 궤적 분석 (Latent Trajectory Analysis)**
*   **목표**: 반복에 따른 "생각의 정교화"를 수치화.
*   **Entropy Decay**: 각 루프 단계의 Logits 엔트로피가 감소하는지 확인 (불확실성 해소).
*   **State Delta**: 루프 간 Hidden State 변화량 $\|h_{l+1} - h_l\|$이 특정 지점에서 수렴하는지 분석.

---

### 2. SPAK 엔지니어링 지식의 핵심 적용

| 축적된 지식 (Asset) | loopLM 검증 및 구현 적용 | 기대 효과 |
| :--- | :--- | :--- |
| **Blackwell TMA Laws** | 반복 루프 내 가중치 **L2 캐시 고정(Pinning)** | HBM 재로드 제거로 연산 밀도 극대화 |
| **Stability Floor ($-1e20$)** | 수십 회 반복 시 발생할 수 있는 **Logits 폭주 방지** | 심층 재귀 구조에서의 수치적 발산 제어 |
| **Hierarchical Parity** | `L1(Kernel) -> L2(Loop) -> L3(Full Model)` 검증 | 새로운 아키텍처의 수학적 무결성 조기 확보 |
| **Autograd Hybrid Mode** | 학습 시 Autograd 보장, 추론 시 cuTile 반복 가속 | 학습 안정성과 실행 성능의 공존 |

---

### 3. 잠재적 추론 능력(Thinking) 검증 프로토콜

`loopLM`이 실제로 "생각"하고 있는지 판단하기 위한 3대 지표입니다.

1.  **Linear Probing (생각의 깊이 측정)**:
    *   각 루프 단계($l=1 \dots 12$)의 출력에 고정된 분류기를 붙여 정답을 얼마나 일찍 맞히는지 측정.
    *   루프가 진행될수록 분류 성능이 향상된다면, 잠재 공간에서 정보가 고도화되고 있음을 의미.

2.  **Fixed-Point Convergence (사고의 멈춤)**:
    *   잠재 표현의 변화량이 임계값($\epsilon$) 이하로 떨어지는 지점을 측정.
    *   복잡한 문장 구조에서 더 늦게 수렴한다면 모델이 난이도를 인지하고 "더 깊이 생각"하는 것임.

3.  **Cross-Architecture Baseline**:
    *   `loopLM`과 동일한 파라미터 수를 가진 `GPT-1L` (공간적 얕음) 대비 성능 우위가 뚜렷한지 확인.

---

### 4. 핵심 엔지니어링 접근법

#### **4.1. 지속적 가중치 캐싱 (Persistent Weights)**
`loopLM`은 동일한 가중치를 반복 호출하므로, `ct.load` 시 `latency` 최적화와 더불어 가중치를 공유 메모리에 상주시키는 전용 커널 설계가 필수적입니다.

#### **4.2. 가변적 깊이 제어 (Adaptive Depth)**
SPAK **Outer Loop**가 잠재 공간의 변화율을 모니터링하여 `break` 시점을 결정하는 동적 루프 로직을 DSL 수준에서 정의합니다.

---

### 💎 결론: 시스템적 진화의 정점

`loopLM` 개발은 단순히 모델을 만드는 과정이 아닙니다. **"Standard GPT(12L)"라는 거인**을 기준으로 세우고, 우리가 NanoGPT에서 배운 **계층적 등가성 검증**과 **Blackwell 하드웨어 법칙**을 결합하여 "적은 자원으로 깊은 사고를 하는" 지능형 시스템을 구축하는 과정입니다. 

우리의 첫 번째 임무는 **표준 12레이어 GPT의 무결성을 완벽히 확보**하는 것에서 시작합니다.

---
✦ 검색 결과와 최신 연구 흐름을 종합하여 loopLM의 아키텍처 및 검증 전략을 한층 더 구체화할 수 있는 핵심 지표들을 확보했습니다. 특히 "Thinking Step Encoding"과 "Adaptive Entropy-based Exit" 개념이 우리 SPAK DSL에 포함되어야 할 결정적 요소임을 확인했습니다.

  ---

  1. loopLM 정밀 설계 포인트 (Refined Architecture)

   1. Thinking Step Encoding (TSE):
       * 단순히 레이어를 반복하는 것이 아니라, 각 루프 단계($l=1 \dots L$)에 대한 고유한 인코딩을 추가하여 모델이 "현재
         몇 번째 사고 단계인지"를 알게 해야 합니다.
       * 이는 SPAK DSL의 tuning_space에 step_encoding_type: ["Temporal_RoPE", "Step_Learned_Bias"]와 같은 항목으로
         정의될 수 있습니다.

   2. Adaptive Token Routing & Exit:
       * 모든 토큰이 동일한 횟수만큼 반복되는 것이 아니라, 엔트로피 임계값($H_{target}$) 이하로 떨어진 토큰은 조기에
         연산을 멈추는 "Early Exit" 로직이 핵심입니다.
       * 이는 Blackwell 하드웨어에서 조건부 실행(Conditional Execution) 부하를 줄이기 위한 "Masked Loop Logic"으로
         구현되어야 합니다.

   3. Input Injection (Residue Link):
       * 각 반복 단계의 입력에 원본 임베딩 $x_0$를 다시 더해주는(Residue) 방식이 잠재 표현의 "망각"을 방지하고 추론을
         안정화하는 데 효과적임을 확인했습니다.

✦ 결론: 엔지니어링 실행 계획 (Revised)

  논문 리뷰와 v2.dsl 설계를 통해 우리가 가야 할 길은 명확해졌습니다.

   1. Baseline 완성: 현재 진행 중인 표준 GPT(12L) 학습을 완결하여 "정답 궤적"을 확보합니다.
   2. Adaptive Kernel 설계: 단순히 for 루프를 돌리는 것이 아니라, 엔트로피 기반 Early Exit이 가능한 cuTile 커널을 설계합니다. (이는 Blackwell 하드웨어에서 엄청난 처리량 이득을 줄 것입니다.)
   3. Thinking Trace 분석: 추론 시 각 루프마다 문장이 어떻게 "더 똑똑해지는지"를 엔트로피 곡선으로 시각화하여 증명합니다.


  주요 DSL 통합 포인트:
   1. Thinking Mechanism 통합: Step Encoding, X0 Residue Injection, Adaptive Early Exit 등 최신 연구(Ouro, ITT)의 핵심
      아키텍처 요소를 design_space에 반영했습니다.
   2. Blackwell 최적화 규칙: 기존 NanoGPT에서 얻은 Stability Floor (-1e20)와 Weight Pinning 전략을 tuning_space와
      knowledge에 명시했습니다.
   3. 검증 프로토콜 정교화: Space-Time Equivalence와 Entropy Decay를 불변성(Invariant)으로 정의하여, 모델이 실제로
      "생각"하고 있음을 증명하는 체계를 구축했습니다.

  이제 이 DSL은 loopLM 프로젝트의 최상위 설계 명세서 역할을 하게 됩니다.