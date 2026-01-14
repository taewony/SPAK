# Research Potential Analysis & Roadmap

## 1. Value Proposition (Publishability)

**Conclusion: High Potential**

Most current Agent research focuses on "Performance" (e.g., SWE-bench scores). This project addresses **"Correctness & Safety"** through Architecture & Process, tackling the critical issues of **Reliability** and **Controllability** in AI Software Engineering (AI4SE). This makes it highly relevant for top-tier conferences like ICSE, FSE, ASE, and NeurIPS.

**Key Contributions:**
1.  **Formal-Spec Driven Synthesis:** A **Neuro-Symbolic approach** controlling LLM output via **Semantic IR (AISpec)**, not just prompts.
2.  **Kernel Architecture:** Managing side-effects via an **Algebraic Effect-based Runtime**, treating agents as controllable software rather than black-box scripts.
3.  **Recursive Verification:** A **Fractal Design Pattern** utilizing a "Spec → Code → Verify → Refine" loop to reliably build complex systems.

---

## 2. Analysis of Prior Art

### A. Multi-Agent Frameworks (Structured Collaboration)
*   **Examples:** **MetaGPT** (ICLR 2024), **ChatDev** (ACL 2024)
*   **Concept:** Waterfall development using SOPs defined in natural language prompts.
*   **The Gap:** They define processes in **Natural Language (Prompt)**; we define them in **Formal DSL (Lark)** for machine verification. They focus on "Collaboration"; we focus on "**Kernel-Level Control (Runtime Safety)**".

### B. Self-Correction & Verification
*   **Examples:** **Reflexion** (NeurIPS 2023), **LDB**, **Clover** (2024)
*   **Concept:** Self-repair loops based on execution traces or error logs.
*   **The Gap:** Existing work mainly checks **Unit Tests**. We combine **Structural Verification (Static Analysis)** with **Behavioral Verification**, checking compliance with **Semantic IR**. We verify "Design Intent," not just "Test Pass."

### C. Formal Methods + LLM
*   **Examples:** **Spec-to-Code** (Microsoft Research), **Baldur** (2023)
*   **Concept:** Generating/Verifying mathematical proofs or formal specs (Dafny, Coq).
*   **The Gap:** Traditional formal methods are too academic. We propose **"Practical Formalism"** using **"Nix-style"** practical specs (YAML/Markdown) accessible to engineers.

---

## 3. Positioning & Titles

**Target Venues:** Software Engineering (SE) conferences or AI Systems tracks.

**Proposed Title 1 (System-focused):**
> **"SBAK: A Formally Verified, Spec-Driven Kernel for Recursive AI Software Synthesis"**

**Proposed Title 2 (Methodology-focused):**
> **"Programming Agents with Specs: Bridging the Gap between Natural Intent and Reliable Execution via Algebraic Effects"**

---

## 4. Research Roadmap

### Step 1: Quantitative Evaluation (Benchmarks)
*   Compare success rates on **SWE-bench** or **HumanEval** using our Spec → Code loop vs. Zero-shot LLM.
*   **Claim:** "Spec-driven verification improves success rate by XX% over raw LLM generation."

### Step 2: Qualitative Evaluation (Reliability)
*   **Side-effect Isolation:** Demonstrate 100% prevention/mocking of destructive actions via Algebraic Effects.
*   **Replayability:** Prove **Deterministic** behavior by passing the same verification vectors on repeated runs.

### Step 3: Ablation Study
*   Performance with vs. without the Verifier.
*   Code quality comparison: Description-only vs. Formal Spec.

---

## 5. Draft Abstract (Target: ICSE)

### **Title: SBAK: A Formally Verified, Spec-Driven Agent Kernel for Recursive AI Software Synthesis**
#### (재귀적 AI 소프트웨어 합성을 위한 형식 검증된 명세 기반 Agent 커널)

**Abstract:**
The rapid advancement of Large Language Models (LLMs) has enabled autonomous software generation, yet current agentic frameworks often suffer from non-determinism, lack of interpretability, and uncontrolled side effects. Existing approaches predominantly rely on prompt engineering or social orchestration (multi-agent chat), which fails to guarantee adherence to rigorous architectural constraints or functional correctness beyond simple unit tests.

In this paper, we introduce the **Spec-Driven Build-Agent Kernel (SBAK)**, a novel architecture that treats AI software synthesis as a formal compilation process rather than probabilistic text generation. SBAK introduces three key innovations: (1) **Semantic Intermediate Representation (AISpec)**, a domain-specific language (DSL) that formalizes agent intent, interfaces, and invariants separate from implementation; (2) an **Algebraic Effect-based Runtime Kernel**, which isolates the agent's decision-making policy from execution, ensuring complete observability and safety of side effects; and (3) a **Recursive Verification Loop**, where agents self-correct by iteratively refining code against language-agnostic test vectors and static structural analysis.

We evaluate SBAK on a suite of complex software engineering tasks, demonstrating that our spec-driven approach significantly outperforms standard zero-shot and chain-of-thought methods in generating structurally correct and secure code. Furthermore, we show that our kernel's recursive architecture allows for the deterministic synthesis of complex, hierarchical systems, effectively bridging the gap between natural language intent and reliable, verifiable software execution. Our work proposes a shift from "prompting agents" to "programming agent kernels," offering a foundation for the next generation of reliable AI-assisted software engineering.

  The abstract highlights the three key innovations:
   1. Semantic Intermediate Representation (AISpec)
   2. Algebraic Effect-based Runtime Kernel
   3. Recursive Verification Loop