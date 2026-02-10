# SPAK: Semantic Programmable Agent Kernel
## Semiformal DSL-based System Engineering Framework

**"From Prompting to Programming: A Cognitive Compiler for Autonomous Engineering"**

SPAK is a **Programmable Agent Kernel** designed to teach and enforce rigorous engineering practices in AI Agent development. Unlike frameworks that focus on "chaining prompts," SPAK treats Agent Logic as a formal artifact that must be **compiled from a Semiformal DSL**, **verified against Invariants**, and **executed within an Effect-Isolated Runtime**.

This project serves as a reference implementation for **"Systematic Intelligence Engineering"**, providing a structured framework where Semiformal DSL acts as the **Intermediate Representation (IR)** between latent reasoning (LLM) and deterministic execution (Python/CUDA).

---

## 🏗 Core Architecture

> **"An Agent is an Optimizer over a Domain-Specific Language."**

SPAK operates on a Dual-Loop Architecture supported by four fundamental pillars:

1.  **Semiformal DSL as IR:** A formal language (Grammar) that encodes expert knowledge, invariants, and design rules. It acts as the bridge between the probabilistic Outer Loop and the deterministic Inner Loop.
2.  **Dual-Loop Runtime:**
    *   **Outer Loop (Agent):** The "Architect". Uses abductive reasoning to refine the DSL specification.
    *   **Inner Loop (Engineering):** The "Engineer". Executes the DSL specification to produce artifacts and polymorphic traces.
3.  **Polymorphic Traceability:** The Kernel captures execution metrics (Performance), logical proofs (Correctness), and reasoning steps (Hypothesis) as unified **TraceItems**, enabling grounded self-correction.
4.  **Knowledge Crystallization:** Engineering breakthroughs (e.g., "Matrix Swizzling") are "lifted" from code back into the DSL as reusable **Rules**, accumulating a transferable knowledge base.

### **Architectural Overview**

```
┌─────────────────────────────────────────────────────────────┐
│                   Meta-Level Supervision                     │
│  • System improvement analysis                              │
│  • Specification evolution (DSL v1 -> v2)                   │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│          Systematic Intelligence Engineering (SIE)          │
│  ┌─────────────────────────────────────────────────────┐  │
│  │     Autonomous Engineering System (AES)             │  │
│  │  ┌─────────────────────────────────────────────┐   │  │
│  │  │    Outer Loop: Reasoning (LLM + DSL IR)     │   │  │
│  │  │  (SPAK Kernel: Compiler & Planner)          │   │  │
│  │  └───────────────────┬─────────────────────────┘   │  │
│  │                      │ (Artifacts & Scripts)       │  │
│  │  ┌───────────────────▼─────────────────────────┐   │  │
│  │  │    Inner Loop: Execution (Python/CUDA)      │   │  │
│  │  │  (Trace Generation & Validation)            │   │  │
│  │  └─────────────────────────────────────────────┘   │  │
│  └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 📚 The Loop Taxonomy

SPAK recognizes that not all execution loops are equal. It formalizes three distinct types of loops within the DSL:

| Loop Type | Role | Determinism | Latency | Reasoning (LLM) | Typical Use |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Agent Loop** | **Strategist** | Low (Probabilistic) | High (Seconds) | **Yes** (Heavy) | Planning, Debugging, Architecture Design |
| **Service Loop** | **Operator** | High (Strict) | Low (Milliseconds) | **No** (Logic only) | API Serving, Request Handling, Event Routing |
| **Engineering Loop** | **Experimenter** | High (Repeatable) | Variable (Minutes) | **No** (Search only) | Auto-tuning, Benchmarking, A/B Testing |

---

## 🛠 Usage & Workflow

The system runs a **Trace-Guided Engineering Loop**.

### 1. Define the System (DSL)
Create a `.dsl` file to define your domain model, invariants, and optimization rules.

```spak-dsl
system MatMul_Optimizer {
    knowledge {
        fact is_memory_bound(t) { return t.utilization < 0.6 }
        rule "Tiling" {
            when: "is_memory_bound"
            apply: "Increase tile size to maximize L2 reuse."
        }
    }
    engineering_loop Tuner { ... }
}
```

### 2. Execute the Engineering Loop
The Kernel compiles the DSL into executable Python scripts (the "Service Loop") and runs them.

```bash
# Generate artifacts and run the optimization cycle
$ python spak_v2.py run examples/matmul_system_v2.dsl
```

### 3. Verify via Grounded Traces
The execution produces structured `__SPAK_TRACE__` JSON logs. The Outer Loop analyzes these against the DSL's `trace_schema`.

```json
{
  "type": "Performance",
  "step_name": "Level 5: Auto-Tuned",
  "tflops": 67.37,
  "speedup": 0.98
}
```

If the trace satisfies the **Success Criteria** defined in the DSL, the artifact is accepted. If not, the Agent applies the next Abductive Rule from the Knowledge block.

---

## 🌍 Why "Kernel"?

In Computer Science, a **Kernel** manages resources, provides abstraction, and enforces isolation. SPAK does exactly this for AI Engineering:

1.  **Cognitive Resource Management:** Separates high-cost reasoning (Outer Loop) from low-cost execution (Inner Loop).
2.  **Abstraction Layer:** Provides a **Semiformal DSL** as a stable interface for defining complex systems, independent of the underlying implementation language (Python, CUDA, etc.).
3.  **Knowledge Accumulation:** Acts as a persistent store for engineering "Rules" and "Invariants," preventing the Agent from relearning basics on every run.

---

## 📝 Citation

If you use SPAK for research or education, please cite:

> **"SPAK: A Dual-Loop Cognitive Architecture for Systematic High-Performance Computing Engineering"** (Draft, 2026)