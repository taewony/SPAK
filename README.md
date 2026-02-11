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

Here is the breakdown of why this DSL-as-IR model is transformative:

  1. The Analogy: SPAK vs. LLVM


  ┌──────────────┬───────────────────────────────────────────┬─────────────────────────────────────────────────────┐
  │ Component    │ Standard Compiler (LLVM)                  │ SPAK Agent Framework                                │
  ├──────────────┼───────────────────────────────────────────┼─────────────────────────────────────────────────────┤
  │ Source       │ C++ / Rust (High Level)                   │ User Intent / Expert Know-How (Natural Language)    │
  │ Frontend     │ Clang (Parser)                            │ Encoding Process (Outer Loop / Architect)           │
  │ IR           │ LLVM IR (Platform Independent)            │ Semiformal DSL (Concept Independent)                │
  │ Optimization │ Passes (Dead Code Elimination, Unrolling) │ Reasoning Passes (Trace Analysis, Rule Application) │
  │ Backend      │ x86 / ARM Machine Code                    │ Executables (Python, CUDA, Shell Scripts)           │
  └──────────────┴───────────────────────────────────────────┴─────────────────────────────────────────────────────┘


  2. The Power of "Encoding" (Expert -> DSL)

  You mentioned "Encoding experts' know-how." In traditional software, this is done via comments or documentation, which
  computers cannot execute. In SPAK:

   * Pattern Recognition: Instead of an LLM guessing how to optimize a kernel every time, we encode the specific
     technique (e.g., "Swizzling" or "Double Buffering") into a reusable rule in the DSL.
   * Lossless Compression: The DSL strips away the syntactic noise of Python/C++ (brackets, imports, types) and
     preserves only the Architectural Decisions.
   * Accumulation: Every time the agent solves a problem (like FMHA), it generates a DSL artifact. This file becomes a
     "module" in our Knowledge Library. The next time we need attention, we don't "re-invent" it; we "link" it.

  3. The Power of "Decoding" (DSL -> Executables)

  This is where the Transferability you mentioned shines. Because the logic is stored in the IR (DSL), the Backend
  (Implementation) becomes retargetable.

   * Scenario A (NVIDIA):
       * Input: rule { apply: "Tiling" }
       * Decoder: Generates cuda.tile (Python/CUDA) code.
   * Scenario B (AMD/ROCm):
       * Input: rule { apply: "Tiling" } (The DSL does NOT change!)
       * Decoder: Generates Triton or HIP code.
   * Scenario C (CPU):
       * Decoder: Generates numpy logic (for verification).

  The Knowledge (The Design) is preserved in the DSL, while the Implementation is transient and adaptable.

  4. The "Optimization Pass" (Trace-Guided Refinement)

  In a standard compiler, an "Optimization Pass" looks at the IR and improves it.
  In SPAK, the Engineering Loop acts as a dynamic optimization pass:

   1. Run: Execute current IR (DSL).
   2. Profile: Generate TraceItem.
   3. Analyze: Match Trace against Fact (e.g., is_memory_bound).
   4. Transform: Apply Rule (e.g., apply: "Fusion").
   5. Result: A refined IR (DSL v2).

  5. Why this is "Difficult with other approaches"

   * Context Window Limits: Standard Agents try to keep the entire codebase and history in the context window. This
     explodes quickly. DSLs are concise; they compress "100 lines of CUDA" into "3 lines of Logic."
   * Stochastic Drift: If you ask an LLM to "optimize this" 10 times, you get 10 different, random approaches. If you
     ask it to "Apply the Swizzling Rule defined in the DSL," you get deterministic, repeatable engineering.

  Conclusion

  By treating the DSL as an IR for Engineering Knowledge, you transform the Agent from a "Chatbot" into a "System
  Synthesizer."

   * Old Way: LLM writes code directly. (Fragile, not reusable).
   * SPAK Way: LLM writes the Design (DSL) -> The Kernel compiles Design to Code.

  This confirms that your proposed architecture is not just a tool, but a Knowledge Management System for AI
  Engineering.