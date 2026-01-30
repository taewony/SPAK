# FMHA Step 3: Fused & Pipelined Kernel
# Implementation based on cuTile

import cuda.tile as ct
import torch
import math
import numpy as np
from cuda.tile import RoundingMode as RMd

INV_LOG_2 = 1.0 / math.log(2)
ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]

# ============================================================
# 1. Fused Kernel Definition
# ============================================================
@ct.kernel(occupancy=2)
def fmha_kernel(Q, K, V, Out,
                qk_scale: float,
                input_pos: int,
                TILE_D: ConstInt,  # TILE_D = hidden_size
                H: ConstInt,
                TILE_M: ConstInt,
                TILE_N: ConstInt,
                QUERY_GROUP_SIZE: ConstInt,
                CAUSAL: ConstBool,
                EVEN_K: ConstBool):
    """
    cuTile kernel for Fused Multi-Head Attention (FMHA).
    Computes attention output for a specific batch item and head, using tiling and online softmax.
    """
    # Map block IDs to batch and head indices
    bid_x = ct.bid(0)
    bid_y = ct.bid(1)
    batch_idx = bid_y // H
    head_idx = bid_y % H
    off_kv_h = head_idx // QUERY_GROUP_SIZE

    # Adjust qk_scale for exp2
    qk_scale = qk_scale * INV_LOG_2

    # Initialize offsets for current query tile (M-dimension)
    offs_m = bid_x * TILE_M + ct.arange(TILE_M, dtype=np.int32)  # [TILE_M]
    offs_m += input_pos
    offs_m = offs_m[:, None]  # [TILE_M, 1]

    # Initialize local offsets for key/value tile (N-dimension)
    offs_n_tile = ct.arange(TILE_N, dtype=np.int32)  # [TILE_N]
    offs_n_tile = offs_n_tile[None, :]  # [1, TILE_N]

    # Initialize online softmax accumulators in float32 for stability
    m_i = ct.full((TILE_M, 1), -np.inf, dtype=np.float32)
    l_i = ct.full((TILE_M, 1), 0.0, dtype=np.float32)
    acc = ct.full((TILE_M, TILE_D), 0.0, dtype=np.float32)

    # Load query tile for this batch, head, and M-chunk
    # Strategy: Load Q once per block (Register/SRAM)
    q = ct.load(
        Q, index=(batch_idx, head_idx, bid_x, 0), shape=(1, 1, TILE_M, TILE_D)
    ).reshape((TILE_M, TILE_D))  # [TILE_M, TILE_D]

    # loop over k, v and update accumulator
    m_end = input_pos + (bid_x + 1) * TILE_M
    k_seqlen = K.shape[2]
    
    # Calculate Loop Count (Tc)
    if CAUSAL:
        mask_start = (input_pos + bid_x * TILE_M) // TILE_N
        mask_start = min(mask_start, k_seqlen // TILE_N)
        Tc = ct.cdiv(min(m_end, k_seqlen), TILE_N)
    else:
        Tc = ct.cdiv(k_seqlen, TILE_N)
        mask_start = k_seqlen // TILE_N

    # Loop over K, V blocks (N-dimension chunks)
    for j in range(0, Tc):
        # --- Compute QK product ---
        # Strategy: Pipelined Load of K (latency=2)
        k = ct.load(
            K, index=(batch_idx, off_kv_h, 0, j), shape=(1, 1, TILE_D, TILE_N),
            order=(0, 1, 3, 2), # Permute K for efficient MMA (K^T)
            latency=2,
        )
        k = k.reshape((TILE_D, TILE_N))  # [TILE_D, TILE_N]
        qk = ct.full((TILE_M, TILE_N), 0., dtype=np.float32)
        
        # Tensor Core Operation
        qk = ct.mma(q, k, qk)  # [TILE_M, TILE_N]

        # --- Apply Causal Masking ---
        if (CAUSAL or not EVEN_K) and j >= mask_start:
            offs_n = j * TILE_N + offs_n_tile
            mask = ct.full((TILE_M, TILE_N), True, dtype=np.bool)
            if not EVEN_K:
                mask = mask & (offs_n < k_seqlen)
            if CAUSAL:
                mask = mask & (offs_m >= offs_n)  # [TILE_M, TILE_N]
            mask = ct.where(mask, 0.0, -np.inf)  # [TILE_M, TILE_N]
            qk += mask

        # --- Online Softmax Update ---
        m_ij = max(m_i, ct.max(qk, axis=-1, keepdims=True) * qk_scale)
        qk = qk * qk_scale - m_ij  # [TILE_M, TILE_N]

        # attention weights
        p = ct.exp2(qk, flush_to_zero=True)  # [TILE_M, TILE_N]
        l_ij = ct.sum(p, axis=-1, keepdims=True)  # [TILE_M, 1]
        alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)  # [TILE_M, 1]
        
        # Update State
        l_i = l_i * alpha + l_ij  # [TILE_M, 1]
        acc = acc * alpha  # [TILE_M, TILE_N]

        # --- Compute PV product ---
        # Strategy: Pipelined Load of V (latency=4)
        v = ct.load(
            V, index=(batch_idx, off_kv_h, j, 0), shape=(1, 1, TILE_N, TILE_D),
            latency=4,
        ).reshape((TILE_N, TILE_D))  # [TILE_N, TILE_D]
        
        p = p.astype(Q.dtype)
        acc = ct.mma(p, v, acc)  # [TILE_M, TILE_N]
        m_i = m_ij  # [TILE_M, 1]

    # --- Final Normalization and Store ---
    acc = ct.truediv(acc, l_i, flush_to_zero=True, rounding_mode=RMd.APPROX)
    acc = acc.reshape((1, 1, TILE_M, TILE_D)).astype(Out.dtype)
    ct.store(Out, index=(batch_idx, head_idx, bid_x, 0), tile=acc)


# ============================================================
# 2. Wrapper / Main
# ============================================================
def main():
    print("=== FMHA Step 3: Fused Kernel (Implementation) ===")
    
    # Configuration
    BATCH_SIZE = 2
    NUM_HEADS = 8
    SEQ_LEN_Q = 128
    SEQ_LEN_KV = 128
    D_K = 64
    D_V = 64
    TILE_M = 128
    TILE_N = 128
    QUERY_GROUP_SIZE = 1
    DTYPE = torch.float16
    
    # 1. Setup Data
    Q = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, D_K, dtype=DTYPE, device='cuda')
    K = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_KV, D_K, dtype=DTYPE, device='cuda')
    V = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_KV, D_V, dtype=DTYPE, device='cuda')
    Out = torch.empty((BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, D_V), dtype=DTYPE, device='cuda')

    qk_scale = 1.0 / math.sqrt(D_K)
    grid = (math.ceil(SEQ_LEN_Q / TILE_M), BATCH_SIZE * NUM_HEADS, 1)

    print(f"Launching Kernel (Grid: {grid})...")
    
    # 2. Launch
    ct.launch(torch.cuda.current_stream(), grid, fmha_kernel, (
        Q, K, V, Out,
        qk_scale,
        0, # input_pos
        D_K,
        NUM_HEADS,
        TILE_M,
        TILE_N,
        QUERY_GROUP_SIZE,
        False, # causal
        True   # even_k (128 % 128 == 0)
    ))

    # 3. Verify
    print("Verifying...")
    ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=False)
    
    try:
        torch.testing.assert_close(Out, ref, atol=1e-2, rtol=1e-2)
        print("✅ Verification: Success!")
    except Exception as e:
        print(f"❌ Verification: Failed! {e}")

if __name__ == "__main__":
    main()