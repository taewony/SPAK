import math
import torch
import cuda.tile as ct
import numpy as np

# Define type aliases
ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]

# ============================================================
# 1. LoopLM Optimized Attention Kernel (4D Tiling Law)
# ============================================================

@ct.kernel(occupancy=2)
def looplm_attention_kernel(
    Q, K, V, Out,
    qk_scale_log2: float, neg_inf: float,
    TILE_D: ConstInt, H: ConstInt,
    TILE_M: ConstInt, TILE_N: ConstInt,
    K_LAT: ConstInt, V_LAT: ConstInt
):
    bid_x, bid_y = ct.bid(0), ct.bid(1)
    batch_idx, head_idx = bid_y // H, bid_y % H
    
    m_i = ct.full((TILE_M, 1), neg_inf, dtype=ct.float32)
    l_i = ct.full((TILE_M, 1), 0.0, dtype=ct.float32)
    acc = ct.full((TILE_M, TILE_D), 0.0, dtype=ct.float32)

    offs_m = bid_x * TILE_M + ct.arange(TILE_M, dtype=ct.int32)[:, None]
    offs_n_tile = ct.arange(TILE_N, dtype=ct.int32)[None, :]

    # 4D LAW: index uses tile-unit (bid_x)
    q = ct.load(Q, index=(batch_idx, head_idx, bid_x, 0), shape=(1, 1, TILE_M, TILE_D), padding_mode=ct.PaddingMode.ZERO).reshape((TILE_M, TILE_D))
    
    k_seqlen = K.shape[2]
    m_end = (bid_x + 1) * TILE_M
    Tc = ct.cdiv(min(m_end, k_seqlen), TILE_N)

    for j in range(0, Tc):
        # 4D LAW: index uses tile-unit (j)
        k = ct.load(K, index=(batch_idx, head_idx, 0, j), shape=(1, 1, TILE_D, TILE_N), order=(0,1,3,2), latency=K_LAT, padding_mode=ct.PaddingMode.ZERO).reshape((TILE_D, TILE_N))
        qk = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
        qk = ct.mma(q, k, qk)

        offs_n = j * TILE_N + offs_n_tile
        mask = (offs_m >= offs_n) & (offs_n < k_seqlen)
        qk = qk + ct.where(mask, 0.0, neg_inf)

        m_ij = max(m_i, ct.max(qk, axis=-1, keepdims=True) * qk_scale_log2)
        p = ct.exp2(qk * qk_scale_log2 - m_ij, flush_to_zero=True)
        l_ij = ct.sum(p, axis=-1, keepdims=True)
        alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)
        
        l_i = l_i * alpha + l_ij
        acc = acc * alpha

        # 4D LAW: index uses tile-unit (j)
        v = ct.load(V, index=(batch_idx, head_idx, j, 0), shape=(1, 1, TILE_N, TILE_D), latency=V_LAT, padding_mode=ct.PaddingMode.ZERO).reshape((TILE_N, TILE_D))
        acc = ct.mma(p.astype(Q.dtype), v, acc)
        m_i = m_ij

    acc = ct.truediv(acc, l_i)
    # 4D LAW: index uses tile-unit (bid_x)
    ct.store(Out, index=(batch_idx, head_idx, bid_x, 0), tile=acc.reshape((1, 1, TILE_M, TILE_D)).astype(Out.dtype))

# ============================================================
# 2. LoopLM Halting Kernel (2D Offset Law)
# ============================================================

@ct.kernel
def looplm_halt_update_kernel(
    H_current,    # (M, TILE_N)
    H_next,       # (M, TILE_N)
    Logits,       # (M, TILE_V)
    Active_Mask,  # (M, 1)
    Steps_Taken,  # (M, 1)
    Threshold: float,
    TILE_SIZE_M: ConstInt,
    TILE_SIZE_N: ConstInt,
    TILE_SIZE_V: ConstInt
):
    bid = ct.bid(0)
    # 2D LAW: index must use EXPLICIT ELEMENT OFFSET (bid * TILE_SIZE_M)
    # This is because these tensors are treated as flat row-major pools
    h_curr = ct.load(H_current, index=(bid * TILE_SIZE_M, 0), shape=(TILE_SIZE_M, TILE_SIZE_N))
    h_next = ct.load(H_next, index=(bid * TILE_SIZE_M, 0), shape=(TILE_SIZE_M, TILE_SIZE_N))
    mask = ct.load(Active_Mask, index=(bid * TILE_SIZE_M, 0), shape=(TILE_SIZE_M, 1))
    steps = ct.load(Steps_Taken, index=(bid * TILE_SIZE_M, 0), shape=(TILE_SIZE_M, 1))
    
    logits = ct.load(Logits, index=(bid * TILE_SIZE_M, 0), shape=(TILE_SIZE_M, TILE_SIZE_V), padding_mode=ct.PaddingMode.ZERO)
    
    # Calculate Max Probability correctly
    max_logit = ct.max(logits, axis=1, keepdims=True)
    exp_logits = ct.exp(logits - max_logit)
    sum_exp = ct.sum(exp_logits, axis=1, keepdims=True)
    max_prob = 1.0 / sum_exp
    
    was_active = mask > 0.5
    is_active_next = (max_prob < Threshold) & was_active
    
    # Update and Store
    h_updated = ct.where(was_active, h_next, h_curr)
    mask_updated = ct.astype(is_active_next, ct.float32)
    steps_updated = steps + ct.astype(was_active, ct.int32)
    
    ct.store(H_current, index=(bid * TILE_SIZE_M, 0), tile=h_updated)
    ct.store(Active_Mask, index=(bid * TILE_SIZE_M, 0), tile=mask_updated)
    ct.store(Steps_Taken, index=(bid * TILE_SIZE_M, 0), tile=steps_updated)
