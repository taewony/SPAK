# SPAK Project Plan & Roadmap

## 🎯 Vision: Systematic Compound Engineering

Compound Engineering의 핵심은 개별 작업 단위의 결과가 단순히 산출물에 그치지 않고, 다음 작업을 더 쉽고 효율적으로 만드는 ‘복리 구조(Compounding)’를 구축하는 데 있습니다.
이 과정에서 Semiformal DSL은 도메인 지식을 정형화하고 AI 에이전트와의 협업 효율을 극대화하는 가교(Bridge) 역할을 수행합니다.

1. Semiformal DSL이 Compound Engineering에 효과적인 이유 (Key Benefits)

✅ 지식의 자산화 (Codifying Knowledge)
Compound Engineering은 발생한 버그, 코드 리뷰 의견, 설계 결정 등을 단순 기록을 넘어 시스템의 ‘기억’으로 축적합니다.
Semiformal DSL은 자연어의 모호함을 줄이면서도 완전한 프로그래밍 언어보다 유연하여, 이러한 경험적 지식을 AI가 즉시 이해하고 실행할 수 있는 규칙으로 변환하는 데 최적의 도구입니다.

✅ AI 에이전트의 정밀도 향상
단순 자연어 프롬프트 대신 DSL 구조로 지시를 내리면 AI 에이전트가 생성하는 결과물의 일관성(Consistency)과 예측 가능성(Predictability)이 획기적으로 높아집니다.
이는 실행(Work) 단계에서의 오류를 사전에 차단하고 전체 개발 루프의 속도를 가속화합니다.

✅ 복잡성 제어 및 추상화 (Abstraction & Complexity Management)
시스템 규모가 커질수록 도메인 전문가와 개발자 사이의 인지적 간극이 벌어집니다.
Semiformal DSL은 의미적 브리지(Semantic Bridge) 역할을 하여, 복잡한 비즈니스 로직과 최적화 전략을 간결하고 명확하게 표현하고, 이를 다양한 이해관계자가 공유할 수 있게 합니다.

Semiformal DSL은 단순한 설정 파일이나 명세서가 아닙니다.
Compound Engineering의 핵심 동력으로, 각 사이클에서 얻은 통찰을 기계가 읽고, 인간이 이해하며, 다음 프로젝트에 즉시 재사용할 수 있는 형태로 응축합니다.
이로 인해 조직의 엔지니어링 역량은 선형이 아닌 지수적으로 축적되며, AI 에이전트는 단순한 코드 생성기를 넘어 지식의 전달자이자 확장자로 진화합니다.

---

## 🏗 Current Architecture: Dual-Loop Cognitive System
- **Outer Loop (Agent):** Architect/Strategist. Reason over DSL specifications, analyze traces, and evolve design rules.
- **Inner Loop (Engineering):** Operator/Experimenter. Execute artifacts, perform auto-tuning, and emit structured `TraceItems`.
- **Knowledge Bridge:** Semiformal DSL encoding Ontology, Invariants, and Transformation Rules.

---

## 🚀 Roadmap

### Phase 1-3: Core Infrastructure (COMPLETED)
- [x] DSL Grammar v2 (`grammar.lark`) & Compiler.
- [x] Effect-Isolated Runtime with Trace Logging.
- [x] Dual-Loop Control Flow (Agent, Service, Engineering Loops).
- [x] Multi-Backend support (Ollama, local Python).

### Phase 4: Industrial Case Studies (COMPLETED)
- [x] **MatMul Optimization**: Implementation of Tiling, Swizzling, and Pipelining via SPAK.
- [x] **FMHA (Fused Multi-Head Attention)**: Implementation of Online Softmax and Kernel Fusion.
- [x] **Verification**: Performance benchmarking (TFLOPS) and correctness proofs against PyTorch/cuTile.

### Phase 5: Academic Foundation & DSL Lift (CURRENT FOCUS)
**Objective**: Transition from "Engineering Tool" to "Scientific Methodology" for GPU Kernel Design.

#### 5.1 Reverse Engineering Methodology (`CuTile2DSL`)
- [x] **Methodology Development**: Formalize the process of extracting "Design Space" from high-performance implementations.
- [x] **Pattern Definition**: Defined 9+ pattern matchers for `CuTile2DSL` in `specs/cutile_patterns.json`.
- [ ] **Implementation**: Develop a pattern-based static analysis tool to lift implicit design decisions into DSL.
- [ ] **Case Study**: Deep dive into NVIDIA's cuTile FMHA to extract 12+ implicit design axes.

#### 5.2 Experimental Validation (Academic Claims)
- [ ] **Claim 1 (Fidelity)**: Prove that DSL-reconstructed forward kernels match the performance of `attention.py` (including TMA hints).
- [ ] **Claim 2 (Semantic Growth)**: Demonstrate that adding GQA and Training-mode support to the DSL is a "compounding" operation on top of v3.

#### 5.3 DSL Schema Evolution
- [x] Extend `system_model` to support `design_space` and `tuning_space` in `grammar.lark`.
- [x] Created `fmha_system_v3.dsl` with separated design/tuning spaces and semantic layer.
- [ ] **FMHAv4 (Forward-Deep-Dive)**: 
    - [ ] **Axis Extraction**: Lift `GQA` mapping logic and `latency` hints.
    - [ ] **Context Logic**: Formalize the switch between `fmha_kernel` and `fmha_fwd_kernel_with_lse`.
    - [ ] **Robustness**: Encode `EVEN_K` as a design choice for "Fast vs. Safe" kernels.

---

## 🎓 Academic Submission Strategy
... (omitted for brevity) ...

---

## 🏗 Dual-PC Compound Engineering Workflow

### Phase 6: Distributed Execution (Current Focus)
**Objective**: Execute the Engineering Loop on the RTX 5070 node and synchronize insights back to the Conceptual Node.

1.  **Conceptual Node (This PC)**:
    *   Maintain and evolve `fmha_system_v4.dsl`.
    *   Generate `fmha_v4_test.py` and `fmha_v4_autotuner.py`.
2.  **Execution Node (RTX 5070 PC)**:
    *   Run `fmha_v4_autotuner.py`.
    *   Capture the `__SPAK_TRACE__` JSON output.
3.  **Synchronization**:
    *   Paste the trace results back to the Conceptual Node.
    *   **Compound Step**: Update `fmha_system_v4.dsl` knowledge base with finalized "Optimal Facts".

---

## 🛠 Active Task List (Immediate Actions)
- [x] **Generate V4 Kernel**: Created `fmha_v4_test.py` with GQA and TMA latency support.
- [x] **Draft Autotuner**: Created `fmha_v4_autotuner.py` to sweep TMA parameters.
- [ ] **Transfer & Run**: User executes `fmha_v4_autotuner.py` on RTX 5070 PC.
- [ ] **Bridge Insights**: Integrate execution results into `fmha_system_v4.dsl` as `fact` or `rule`.
- [ ] **Finalize V4**: Close the loop by updating the "Fidelity" status.

