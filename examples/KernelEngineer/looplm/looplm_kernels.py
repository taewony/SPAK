import math
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

@ct.kernel
def looplm_fused_mlp_halt_kernel(
    H_state,      # (M, TILE_N) - Hidden state
    H_residual,   # (M, TILE_N) - Residual connection (x0 or prev h)
    W_mlp_up,     # (TILE_N, TILE_4N) - MLP Up-projection
    W_mlp_down,   # (TILE_4N, TILE_N) - MLP Down-projection
    W_head,       # (TILE_N, TILE_V) - LM Head for halting
    Active_Mask,  # (M, 1)
    Steps_Taken,  # (M, 1)
    Threshold,    # (M, 1)
    TILE_SIZE_M: ConstInt,
    TILE_SIZE_N: ConstInt,
    TILE_SIZE_V: ConstInt,
    REAL_V: ConstInt
):
    bid = ct.bid(0)
    
    # 1. Load State and Weights (Weights will be persistent in L2)
    h = ct.load(H_state, index=(bid, 0), shape=(TILE_SIZE_M, TILE_SIZE_N))
    res = ct.load(H_residual, index=(bid, 0), shape=(TILE_SIZE_M, TILE_SIZE_N))
    mask = ct.load(Active_Mask, index=(bid, 0), shape=(TILE_SIZE_M, 1))
    
    # 2. State Update (Residual + Projection)
    # Phase 4.1: h_next = h + residual (fused in-place)
    h_next = h + res
    
    # 3. Halting Decision (Fused MatMul with Persistent Head Weights)
    logits = ct.matmul(h_next, W_head)
    max_logit = ct.max(logits, axis=1, keepdims=True)
    is_real_row = max_logit > -1e10
    
    col_idx = ct.arange(TILE_SIZE_V, dtype=ct.int32)[None, :]
    is_real_col = col_idx < REAL_V
    
    logits_centered = logits - ct.where(is_real_row, max_logit, 0.0)
    exp_logits = ct.exp(logits_centered)
    sum_exp = ct.sum(ct.where(is_real_row & is_real_col, exp_logits, 0.0), axis=1, keepdims=True)
    
    max_prob = ct.where(is_real_row & (sum_exp > 0), 1.0 / sum_exp, 0.0)
    
    was_active = mask > 0.5
    is_active_next = (max_prob < threshold) & was_active
    
    # 4. Persistence Store
    ct.store(H_state, index=(bid, 0), tile=ct.where(was_active, h_next, h))
    ct.store(Active_Mask, index=(bid, 0), tile=ct.astype(is_active_next, ct.float32))
    ct.store(Steps_Taken, index=(bid, 0), tile=steps + ct.astype(was_active, ct.int32))
