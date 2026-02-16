# Kernel Spec: Flash Multi-Head Attention (FMHA) for Blackwell

## 1. Metadata & Hardware Truths
* **Target Device**: NVIDIA RTX 5070 (Blackwell Architecture)
* **Backend**: `cuda.tile` (cuTile), `cuda.tile_experimental`
* **Precision**: `bfloat16` (Compute & IO), `float32` (Accumulation)
* **Constraint**: Use `tma` (Tensor Memory Accelerator) hints where possible.
* **Optimization**: Requires `autotune` for Tile sizes (M, N).

## 2. Mathematical Definition
Compute the scaled dot-product attention with Causal Masking.

$$
O = \text{Softmax}(Q K^T \cdot \text{scale} + \text{Mask}) V
$$

* **Logic**: Online Softmax (Safe Softmax) for numerical stability.
* **Variant**: Grouped Query Attention (GQA) supported.
    * `num_heads_q` % `num_heads_kv` == 0
    * `query_group_size` = `num_heads_q` // `num_heads_kv`

## 3. Kernel Interfaces (L0 Definition)

### A. Forward Kernel (Inference Optimized)
* **Goal**: Maximize throughput, no need to store intermediate states.
* **Inputs**: `Q`, `K`, `V`, `scale`, `input_pos` (for cache aware), `causal` (bool)
* **Outputs**: `O` (Output)
* **Tiling Strategy**:
    * Parallelize over Batch & Heads.
    * Loop over Q tiles (M-dimension) and K/V tiles (N-dimension).
    * Use `ct.mma` for matrix multiplication.

### B. Forward Kernel (Training / Backward Support)
* **Goal**: Compute Output AND store LSE (Log-Sum-Exp) for backward pass.
* **Inputs**: Same as above + `LSE` (Output Buffer)
* **Logic Extension**:
    * Compute $m$ (max) and $l$ (sum exp) during online softmax.
    * Final LSE = $m + \log_2(l)$
    * **Crucial**: Store `LSE` as a 1D flattened tensor using `ct.scatter` because `ct.store` with tile indexing can be tricky for non-contiguous LSE layouts.

### C. Backward Logic (The Hard Truths)
Backward pass requires re-computation and specific gradient flows.
Input: `Q`, `K`, `V`, `O`, `dO`, `LSE`
Output: `dQ`, `dK`, `dV`

#### C-1. Preprocess Kernel
* **Compute**: $\Delta = \text{RowSum}(O \odot dO)$
* **Reason**: Needed for $softmax$ gradient.
* **Truth**: Compute this separately to reduce register pressure in the main backward kernel. Store as 1D flattened.

#### C-2. dK / dV Kernel
* **Parallelism**: One block per KV tile. Loop over Query tiles.
* **GQA Handling**:
    * Since multiple Query heads share one KV head, gradients from all Query heads in the group MUST be accumulated into `dK`, `dV`.
    * Loop `qh_offset` from 0 to `query_group_size`.
* **Math**:
    * $P = \exp(QK^T - LSE)$
    * $dV += P^T \cdot dO$
    * $dK += (P \cdot (dP - \Delta))^T \cdot Q$

#### C-3. dQ Kernel
* **Parallelism**: One block per Query tile. Loop over KV tiles.
* **Math**:
    * $dQ += (P \cdot (dP - \Delta)) \cdot K$

## 4. Implementation Constraints (Code Generation Rules)
1.  **Imports**:
    * `import cuda.tile as ct`
    * `import cuda.tile_experimental as ct_experimental`
    * `from cuda.tile import RoundingMode as RMd`
2.  **Decorators**: Use `@ct.kernel(occupancy=2)` for all kernels.
3.  **Memory Access**:
    * Use `ct.load` with `latency` hints (2 or 4) for global memory loads.
    * Use `flush_to_zero=True` for `exp2` and `truediv`.
4.  **Autotuning**:
    * Implement `cutile_autotune_fmha` wrapper.
    * Allow disabling autotune via `os.environ["DISABLE_AUTOTUNE"]` for fast CI/Debugging.