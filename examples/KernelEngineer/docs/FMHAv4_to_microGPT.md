# FMHAv4에서 MicroGPT로: 지식의 전이와 시스템 통합 (Atom to Molecule)

FMHAv4가 단일 커널의 극한 성능을 추구한 "원자(Atom)" 레벨의 엔지니어링이었다면, MicroGPT는 이를 활용해 전체 학습 루프를 가속화한 "분자(Molecule)" 레벨의 **지식 전이(Knowledge Portability)** 사례입니다.

| 엔지니어링 단계 | **FMHAv4 (Atomic Kernel)** | **MicroGPT (Integrated Molecule)** |
| :--- | :--- | :--- |
| **시스템 범위** | 단일 Attention 연산 (Q, K, V -> O) | Full GPT Block (RMSNorm, Attention, MLP, Residual) |
| **입력 형태** | 고정된 텐서 벤치마크 | 동적인 텍스트 데이터 시퀀스 (names.txt) |
| **핵심 성과** | 135 TFLOPS (하드웨어 한계 돌파) | **142.5x Speedup** (스칼라 대비 압도적 가속) |
| **지식 활용** | 새로운 법칙 발견 (TMA Law) | 발견된 법칙의 **재사용 및 이식** |
| **검증 지표** | 수치적 오차 (Numerical Error) | **손실값 수렴 (Loss Convergence)** |
| **DSL 역할** | 최적화 공간 정의 | **시스템 전이 규칙(Transformation Rules)** 정의 |

---

### 🔷 FMHAv4의 유산: 검증된 "설계 자산"

MicroGPT는 바닥부터 다시 시작하지 않고, FMHAv4에서 확보한 **Blackwell 최적화 자산**을 그대로 상속받았습니다.

*   **하드웨어 법칙 상속**: RTX 5070에서 검증된 `tile_m: 64` 제약과 `V_Lat=5` 파이프라이닝 설정을 그대로 적용하여 시행착오를 제거했습니다.
*   **수치적 안정성 (Stability Floor)**: FMHAv4에서 발견한 `-1e20` 세이프티 플로어(Safety Floor)를 이식하여, FP16 학습 시 발생할 수 있는 NaN 문제를 사전에 차단했습니다.
*   **커널 템플릿**: FMHAv4의 GQA/Causal 지원 로직을 MicroGPT의 어텐션 엔진으로 직접 통합했습니다.

### 🔶 MicroGPT의 혁신: "스칼라에서 타일로"의 도약

Andrej Karpathy의 `microgpt.py`(순수 파이썬/스칼라 오토그라드)를 고성능 GPU 코드로 변환하며 시스템적 복잡성을 해결했습니다.

*   **의미론적 리프팅 (Semantic Lifting)**: 스칼라 루프 기반의 연산(`sum(wi * xi)`)을 `cuTile`의 `ct.mma` 텐서 연산으로 추상화하여 구현했습니다.
*   **커널 퓨전 (Fusion Expansion)**: Attention뿐만 아니라 RMSNorm과 Linear 연산을 결합하고, `RMSNorm`의 스케일 계산을 다음 연산의 로드 단계에 통합하는 최적화 규칙을 수립했습니다.
*   **학습 루프 통합**: 단일 커널 벤치마크를 넘어, 옵티마이저(Adam), 데이터 로더, 그라디언트 업데이트가 포함된 **실제 학습 파이프라인**에서의 성능과 정확도를 증명했습니다.

---

### 📈 원자에서 분자로의 진화 경로

이 단계는 **단일 기술이 어떻게 제품(System)으로 녹아드는지**를 보여주는 핵심 경로입니다.

| 단계 | 과정 명칭 | 핵심 내용 | 복리 효과 (Compounding) |
| :--- | :--- | :--- | :--- |
| **1** | **Inheritance** | FMHAv4의 RTX 5070 최적화 매개변수 로드 | 튜닝 시간 90% 단축 |
| **2** | **Porting** | `microgpt.py`의 논리를 `cuTile` 그래프로 변환 | 가독성 있는 고성능 코드 생성 |
| **3** | **Verification** | `names.txt` 데이터를 통한 손실값 수렴 확인 | 수학적/논리적 무결성 증명 |
| **4** | **Crystallization** | `microgpt_system_v1.dsl`에 전이 규칙 기록 | 다음 프로젝트(NanoGPT)의 기반 마련 |

**진화의 핵심**: MicroGPT 엔지니어링은 **"이동 가능한 지식(Portable Knowledge)"**의 가치를 증명했습니다. FMHAv4에서 고생하며 얻은 "Blackwell의 비밀"이 MicroGPT에서는 당연한 "기본 설정"이 되었고, 이를 통해 엔지니어는 커널 최적화가 아닌 **시스템 아키텍처와 학습 안정성**에 더 집중할 수 있게 되었습니다.

### 💎 결론: 복리의 실현

*   **FMHAv4**는 지식을 **생산**했습니다.
*   **MicroGPT**는 그 지식을 **소비하고 확장**했습니다.

이 과정에서 SPAK DSL은 두 프로젝트 사이의 **지식 브리지(Knowledge Bridge)** 역할을 수행하며, 엔지니어링 통찰이 휘발되지 않고 시스템의 일부로 누적되는 "복리 구조"를 완성했습니다.
