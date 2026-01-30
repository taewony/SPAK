# SPAK Engineering Loop Framework
**The Dual-Loop Architecture for High-Performance Kernel Engineering**

## 1. Conceptual Overview

This framework defines a **Dual-Loop System** for engineering high-performance software in constrained environments. It separates **Strategic Reasoning** (Abductive/Architectural search) from **Tactical Execution** (Inductive/Parameter search).

This separation allows a "Brain" node (CPU-only, low-cost, high-reasoning) to orchestrate a "Muscle" node (GPU-equipped, high-cost, execution-focused) efficiently, minimizing context switching and maximizing hardware utilization.

---

## 2. The Dual Loops

### Loop A: The Outer Loop (Strategic Planner)
*   **Agent Role:** Architect / Physicist.
*   **Environment:** CPU-only (Windows/Local).
*   **Reasoning Mode:** **Abductive** (Inference to the Best Design).
*   **Primary Output:** **Engineering Instruction Guides** (Specs, Invariants, Simulations).
*   **Goal:** Restrict the infinite search space of "code" to a narrow, high-probability "solution valley."

**Responsibilities:**
1.  **Problem Formulation:** Deconstruct complex math (e.g., Attention) into engineering components.
2.  **Invariant Verification:** Write bit-exact Python simulations (e.g., `fmha_step3_fused_sim.py`) to prove the *logic* holds before touching hardware.
3.  **Hypothesis Generation:** "If we fuse loops X and Y, bandwidth usage drops by 50%."
4.  **Instruction Generation:** Create the specific prompt/guide for the Inner Loop (e.g., "Implement this Sim logic using `cuda.tile` primitives").

### Loop B: The Inner Loop (Tactical Optimizer)
*   **Agent Role:** Engineer / Tuner.
*   **Environment:** GPU-accelerated (Linux/Cluster).
*   **Reasoning Mode:** **Inductive/Deductive** (Optimization & Verification).
*   **Primary Output:** **Ground Truth** (TFLOPS, Error Logs, Best Config).
*   **Goal:** Find the global maximum within the "solution valley" defined by the Outer Loop.

**Responsibilities:**
1.  **Implementation:** Translate the *Instruction Guide* (Sim) into hardware-specific code (CUDA/cuTile).
2.  **Compilation & Debugging:** Fix static types, shape mismatches, and syntax errors.
3.  **Auto-Tuning:** Sweep parameters (Tile Sizes, Warps) to find the hardware sweet spot (e.g., `fmha_step4_autotuner.py`).
4.  **Reporting:** Return structured metrics to the Outer Loop to confirm or refute the architectural hypothesis.

---

## 3. The Abductive Reasoning Connection

How does this relate to the "Search for the Optimal Solution"?

### The Engineering Search Space
Imagine a landscape of all possible programs. Most are broken (Error), some are slow (Naive), and few are optimal (FlashAttention). Brute-force searching this is impossible.

### Step 1: Abductive Leap (Outer Loop)
*   *Observation:* "Global Memory access is 100x slower than Compute."
*   *Abductive Inference:* "Therefore, the optimal solution **must** be one that minimizes Global Memory writes."
*   *Result:* The Outer Loop jumps to a specific region of the map: **"The Fused Kernel Region."** It ignores all non-fused solutions. This is the **Strategy**.

### Step 2: Inductive Climb (Inner Loop)
*   *Context:* We are now in the "Fused Kernel Region."
*   *Inductive Action:* "Try 64x64 tiles. Try 128x64 tiles."
*   *Result:* The Inner Loop measures points within this region to find the peak. "64x64 is faster than 128x64." This is the **Optimization**.

> **Summary:** The Outer Loop selects the "Right Hill" to climb (using Abduction). The Inner Loop climbs to the "Summit" of that hill (using Optimization).

---

## 4. Operational Workflow (The Protocol)

1.  **[CPU] Define Invariant:**
    *   Create `step1_ref.py`.
    *   *Result:* "Online Softmax math is valid."
2.  **[CPU] Simulate Kernel:**
    *   Create `step3_sim.py`.
    *   *Result:* "Tiling logic works in Python. Causal masking is correct."
3.  **[CPU -> GPU] Handoff (The Instruction):**
    *   "Translate `step3_sim.py` into `cuda.tile`. Use the logic verified in the sim."
4.  **[GPU] Implementation:**
    *   Write `step3_kernel.py`.
    *   *Result:* "Kernel compiles. Output matches Sim."
5.  **[GPU] Optimization:**
    *   Run `step4_autotuner.py`.
    *   *Result:* "Best config is 64x64 at 62.5 TFLOPS."
6.  **[GPU -> CPU] Feedback:**
    *   Send `Final_Report.md`.
    *   *Strategic Update:* "Hypothesis confirmed. Fusion provided 5x speedup."

## 5. Case Study: FMHA

| Component | Outer Loop (CPU/Strategy) | Inner Loop (GPU/Tactics) |
| :--- | :--- | :--- |
| **Input** | Math Paper ($O = Softmax(QK^T)V$) | Instruction Guide + Sim Code |
| **Logic** | "We need Online Softmax to fuse loops." | "We need to transpose K for Tensor Cores." |
| **Artifact** | `fmha_step3_fused_sim.py` | `fmha_step3_fused_kernel.py` |
| **Verification** | Logic/Math Correctness | Hardware/Timing Correctness |
| **Outcome** | Valid Algorithm Structure | 62.50 TFLOPS Performance |

---

## 6. Conclusion

The **SPAK Engineering Loop** formalizes the relationship between design and execution. By treating **Instruction Guides** as the bridge between Abductive Planning and Tactical Optimization, we ensure that expensive GPU time is never wasted on searching the wrong part of the solution space.
