# FMHAv1에서 FMHAv4로: 시스템 엔지니어링의 진화 (Foundational to Systematic)

FMHAv1이 "수학적 사상을 하드웨어에 매핑하는 기초적인 퓨전(Fusion)"에 집중했다면, FMHAv4는 **"하드웨어의 미세 특성을 포착하고 이를 지식으로 자산화하는 체계적 복리 엔지니어링(Systematic Compound Engineering)"**으로 진화했습니다.

| 시스템 특성 | **FMHAv1 (Foundational Fusion)** | **FMHAv4 (Systematic Compound)** |
| :--- | :--- | :--- |
| **설계 목표** | Online Softmax 기반 메모리 절약 | 하드웨어 한계 돌파 및 지식 자산화 |
| **핵심 기능** | Q-K-V Fusion, Standard Attention | **GQA**, **Causal Masking**, **TMA Pipelining** |
| **최적화 초점** | 기본 타일링 (64x64) 및 루프 구조 | **Blackwell TMA Latency Laws** ($V_{Lat}=5$) |
| **방법론** | Dual-Loop (Strategic/Tactical) | **Systematic Compound Engineering** (SPAK v2) |
| **DSL의 역할** | 연산 및 제어 흐름 기술 (v1/v2) | **Cognitive IR** (Design/Tuning Space 명시) |
| **성능 (RTX 5070)** | ~113 TFLOPS | **~135 TFLOPS** (PyTorch SDPA 대비 1.11x) |
| **지식의 생명주기** | 커널 구현 후 종료 | **복리 축적 (Compounding)**: NanoGPT로 전이 |

---

### 🔷 FMHAv1: “기초적 퓨전 (Foundational Fusion)”

FMHAv1은 FMHA의 핵심 알고리즘인 FlashAttention을 구현하고, 이를 통해 **HBM 트래픽을 최소화하는 시스템적 구조**를 확립하는 데 주력했습니다.

*   **수학적 안정성**: `Online Softmax`를 도입하여 중간 $S, P$ 행렬 저장 없이 출력을 계산하는 상태 관리(State Management) 기법을 검증했습니다.
*   **시스템 경계**: Q, K, V 로드부터 Softmax, MMA 연산, 출력 저장까지의 파이프라인을 단일 커널로 묶는 **Component-Wise Fusion**을 달성했습니다.
*   **검증 전략**: `Hybrid Simulation` (Python Sim + GPU Real)을 통해 복잡한 타일링 로직의 정확도를 하드웨어 없이도 검증할 수 있는 환경을 구축했습니다.

### 🔶 FMHAv4: “지능형 복리 시스템 (Systematic Compound System)”

FMHAv4는 단순한 성능 개선을 넘어, **엔지니어링 과정에서 얻은 통찰을 시스템의 기억(DSL)으로 승화**시켰습니다.

*   **하드웨어 심층 인지**: RTX 5070(Blackwell)의 **TMA(Tensor Memory Accelerator)** 특성을 분석하여, Causal Mask 환경에서 최적의 메모리 오버랩을 위한 `V_Lat=5`와 같은 구체적인 하드웨어 법칙(Laws)을 도출했습니다.
*   **의미론적 확장 (Semantic Growth)**: 기존 v1의 구조를 파괴하지 않고 GQA(Grouped Query Attention)와 Causal Logic이라는 **"델타 지식(Delta-Knowledge)"**을 DSL에 추가함으로써 시스템 복잡도를 선형적으로 제어했습니다.
*   **지식의 결정화 (Knowledge Crystallization)**: 오토튜닝을 통해 발견한 "Blackwell에서 Tile_M=128은 점유율 붕괴를 초래한다"는 부정적 패턴(Negative Pattern)까지 **Safety Invariant**로 DSL에 기록하여 다음 설계의 오류를 사전에 차단합니다.

---

### 📈 v1에서 v4로의 시스템 진화 경로

이 진화 과정은 **수학적 동등성을 넘어, 시스템이 하드웨어와 어떻게 상호작용하며 똑똑해지는지**를 보여줍니다.

| 단계 | 버전 명칭 | 핵심 진화 포인트 | 복리 효과 (Compounding) |
| :--- | :--- | :--- | :--- |
| **1** | **FMHAv1** | Online Softmax & Basic Fusion | FMHA 구현의 기초 아키텍처 확립 |
| **2** | **FMHAv2/3** | GQA 및 Causal Masking 도입 | 다양한 어텐션 변형에 대한 대응력 확보 |
| **3** | **FMHAv4** | **Blackwell TMA 최적화** | 하드웨어 특화 최적화 법칙(Laws) 발견 |
| **4** | **SPAK v2** | **NanoGPT 전이 (Transfer)** | FMHAv4의 지식을 활용해 NanoGPT 2.64x 가속 |

**진화의 핵심**: FMHAv1이 "돌아가는 코드"를 만드는 데 집중했다면, FMHAv4는 그 코드가 **"왜 빠른지(혹은 왜 느린지)"에 대한 이유를 Semiformal DSL로 정형화**했습니다. 이렇게 응축된 지식은 `MicroGPT`를 거쳐 `NanoGPT`라는 더 복잡한 시스템(Organism)을 구축할 때 기초 자산이 되어, 전체 개발 속도를 지수적으로 가속화했습니다.

### 💎 결론: 코드에서 지능으로

*   **FMHAv1**은 우수한 **커널 아키텍처**입니다.
*   **FMHAv4**는 스스로 진화하는 **엔지니어링 지능(Engineering Intelligence)**의 결정체입니다.

FMHAv4의 성공은 단순한 TFLOPS 수치보다, **"한 번 해결한 문제는 다시는 같은 비용을 들여 고민하지 않는다"**는 복리 엔지니어링의 원칙을 실현했다는 점에 그 본질적 가치가 있습니다.
