# FMHA Engineering Plan for RTX 5070

**Objective:** Implement a high-performance Fused Multi-Head Attention (FMHA) kernel optimized for the NVIDIA RTX 5070 architecture (Ampere/Blackwell lineage).
**Constraint:** Move beyond single-kernel optimization (MatMul style) to a component-wise pipeline composition strategy.

## 1. Architectural Strategy: Component-Wise Fusion

Unlike MatMul ($C = A \times B$), FMHA is a stateful pipeline:
$$ O = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d}} + M\right) V $$

We will instruct the inner-loop agent (Kernel Engineer) to build this in 3 strict phases. Verification of *invariants* (mathematical correctness) must precede performance tuning.

### Phase 1: The "Online Softmax" Invariant (State Management)
**Goal:** Verify the agent understands the transition from "Stateless Global Softmax" to "Stateful Online Softmax".
*   **Action:** Implement `OnlineSoftmax` in Python/NumPy first.
*   **Invariant Check:** `OnlineSoftmax([x]) == NaiveSoftmax([x])`
*   **Key Concept:** The tuple `(m, l, acc)` (max, sum, accumulator) is the state that must be preserved across tiles.

### Phase 2: The "Q-K-V" Fusion Block (Pipeline Composition)
**Goal:** Fuse GEMM-I ($S = QK^T$) and GEMM-II ($O = PV$) into a single kernel loop, removing global memory writes for $S$ and $P$.
*   **Action:** Write a `cuda.tile` kernel that:
    1.  Loads a Tile of $Q$ into SRAM/Registers.
    2.  Iterates over Tiles of $K$ and $V$ from global memory.
    3.  Computes $S_{tile} = Q_{tile} \times K_{tile}^T$.
    4.  Updates Softmax State $(m, l, acc)$.
    5.  Computes $O_{acc} += P_{tile} \times V_{tile}$.
*   **Constraint:** Do *not* use Shared Memory for $K$ and $V$ initially; stream them through registers to verify logic.

### Phase 3: The "FlashAttention" Tiling Strategy (Memory Optimization)
**Goal:** Map the logical loop to the RTX 5070's physical hierarchy (L1/Shared Memory).
*   **Hardware Target:** RTX 5070 (Estimated 64KB-100KB+ per SM dynamic shared memory).
*   **Action:**
    1.  **SRAM Budgeting:** Ensure $BlockSize(Q) + BlockSize(K) + BlockSize(V) < SRAM\_Capacity$.
    2.  **Double Buffering:** Apply the pipelining technique verified in `step6_ablation.py` to hide the latency of loading $K$ and $V$ tiles.
    3.  **Swizzling:** Apply the swizzling logic from `step3_swizzling.py` to the Block Grid ($B \times H, M$) to maximize L2 cache reuse of $K/V$ pages across different $Q$ tiles (if applicable) or generally optimize DRAM partition access.

## 2. Execution Roadmap (The "Recipe")

This recipe guides the "Kernel Engineer" agent.

### Step 1: Python Prototype
*   **File:** `fmha_step1_python_ref.py`
*   **Prompt:** "Write a Python function `flash_attention_forward(Q, K, V)` using purely NumPy loops to simulate tiling and online softmax. Verify against `torch.nn.functional.scaled_dot_product_attention`."

### Step 2: Naive Kernel (Baseline)
*   **File:** `fmha_step2_naive_kernel.py`
*   **Prompt:** "Write a `cuda.tile` kernel that implements the logical loop `for tile in K, V:` but without complex double buffering. Focus on correctness of the `m, l, acc` update logic."

### Step 3: Fused & Pipelined Kernel (Target)
*   **File:** `fmha_step3_fused_kernel.py`
*   **Prompt:** "Optimize the Step 2 kernel. 1. Load $Q$ into Shared Memory once. 2. Double-buffer the load of $K$ and $V$. 3. Use `ct.mma` for Tensor Core ops."

### Step 4: Auto-Tuning (Performance)
*   **File:** `fmha_step4_autotuner.py`
*   **Prompt:** "Sweep Tile Sizes ($B_r, B_c$) and Warps/Occupancy. RTX 5070 has high FP16 throughput; prefer larger $B_c$ (64 or 128) if SRAM permits."

## 3. Evaluation Metrics

| Metric | Goal (RTX 5070) | Verification Method |
| :--- | :--- | :--- |
| **Numerical Error** | $< 1e^{-2}$ (FP16) | `torch.allclose(Ref, Out)` |
| **Memory Traffic** | Minimal $S/P$ writes | Profiler / Analytical Model |
| **Throughput** | $> 80\%$ of Theoretical Peak | `TFLOPS` calculation |

## 4. Conclusion
We are building a **Dynamic Context Processor**, not just a matrix multiplier. Correct state management of the Online Softmax is the critical path; performance (Pipelining) is the secondary optimization layer applied only after correctness is proven.
