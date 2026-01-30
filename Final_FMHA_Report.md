# SPAK FMHA Engineering Report
**Date:** 2026-01-30 17:55
**Device:** RTX 5070 (Target)
**Benchmark Size:** B=8, H=16, Seq=1024x1024, D=64

## 1. Executive Summary
This report documents the optimization trajectory of the Fused Multi-Head Attention (FMHA) kernel. The engineering process followed a strict 'Invariant-First' approach, validating mathematical statefulness before optimizing for throughput. The final kernel achieved **113.73 TFLOPS**, a massive speedup over the naive baseline.

## 2. Methodology
The **SPAK Agent** decomposed the optimization into 4 phases, focusing on memory hierarchy efficiency:
*   **Step 1 (Invariant):** Python/NumPy verification of the **Online Softmax** algorithm to ensure numerical stability without storing the full $N \times N$ matrix.
*   **Step 2 (Naive Baseline):** A direct translation of the math to GPU code, writing intermediate Score ($S$) and Probability ($P$) tensors to Global Memory.
*   **Step 3 (Fusion):** Implementing the **Fused Kernel** (FlashAttention style). This keeps $S$ and $P$ in registers/Shared Memory, eliminating the primary memory bottleneck.
*   **Step 4 (Auto-Tuning):** Sweeping tile sizes to find the optimal balance between occupancy and register pressure for the RTX 5070.

## 3. Performance Results

| Step | Strategy | TFLOPS (Measured) | Speedup vs Naive | Status |
|---|---|---|---|---|
| Step 1: Python Prototype | Mathematical Verification (CPU). | - | - | ✅ Pass |
| Step 2: Naive Kernel | Baseline with Global Memory Writes. | **0.27** | **1.00x** | ✅ Pass |
| Step 3: Fused Kernel | Fused QK+PV Pipeline (128x128 Tile). | **38.30** | **141.85x** | ✅ Pass |
| Step 4: Auto-Tuned | Fused Pipeline with Optimal Tile (64x64). | **113.73** | **421.22x** | ✅ Pass |

## 4. Analysis
*   **Naive Bottleneck:** The Naive kernel (Step 2) is severely bound by Global Memory bandwidth. Writing and reading the $O(N^2)$ intermediate matrices ($S, P$) creates massive latency, resulting in < 1 TFLOPS.
*   **The "Fusion" Leap:** Step 3 represents an architectural breakthrough. By fusing the QK and PV loops and applying Online Softmax, we eliminate $O(N^2)$ HBM accesses. This explains the **141x speedup**, shifting the kernel from Memory-Bound to Compute-Bound.
*   **Tuning Impact:** Step 4 shows that while Fusion is the main driver, **Tile Size matters**. The 128x128 tile (Step 3) likely caused register spilling or low occupancy on the RTX 5070. Switching to **64x64** (Step 4) allowed the GPU to hide latency effectively, tripling performance from ~38 to **113 TFLOPS**.

## 5. Conclusion
The FMHA kernel engineering process demonstrates the power of algorithmic innovation over simple code optimization. While MatMul optimization is about "squeezing" performance (e.g., 20-30% gains), FMHA optimization is about **changing the complexity class** of memory access, yielding orders-of-magnitude improvements.