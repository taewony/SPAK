# Root Cause Analysis v3: The "Multiplier Jump" & "Premature Supervision"

**Date**: February 21, 2026
**Conclusion**: cuTile Indexing Law 위반(4배 점프) 및 학습 시 초기 루프 단계의 과도한 Supervision이 지능 발현을 막고 있음.

---

## 1. [HALT BUG] The Indexing Multiplier Error
*   **Problem**: Thinking trace shows 0s for 75% of the sequence (skip-4 pattern).
*   **Cause**: `ct.load(index=(bid * 4, 0), shape=(4, ...))`
*   **Fact**: cuTile `load` with a `shape` argument automatically treats `index` as a **Tile Index**. 
*   **Bug**: Manual multiplication by 4 caused the kernel to skip 3 out of 4 blocks.
*   **Fix**: Use `index=(bid, 0)`. Let cuTile handle the memory stride.

---

## 2. [CONVERGENCE BUG] Premature Supervision Noise
*   **Problem**: Addition loss is flat or increasing.
*   **Cause**: `loss = torch.stack(losses).mean()` over all 12 steps.
*   **Fact**: Arithmetic reasoning requires **Latent Evolution**. Step 1-4 cannot possibly know the carry result of a 4-digit addition.
*   **Bug**: Forcing early steps to match the final target introduces massive gradient noise.
*   **Fix**: **Late-stage Supervision**. Only calculate Loss on the final 4 steps (steps 9-12), allowing steps 1-8 to serve as "Pure Thinking" (Scratchpad).

---

## 3. Systematic Fix Plan
1.  **Kernel**: Remove `* TILE_SIZE_M` from indexing.
2.  **Model**: Change training loss to focus on the final output of the loop.
3.  **Stability**: Ensure `-inf` padding is used correctly to avoid softmax bias.
