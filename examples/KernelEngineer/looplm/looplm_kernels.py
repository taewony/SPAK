import math
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

@ct.kernel
def looplm_halt_update_kernel(
    H_current,    # (M, TILE_N)
    H_next,       # (M, TILE_N)
    Logits,       # (M, TILE_V)
    Active_Mask,  # (M, 1)
    Steps_Taken,  # (M, 1)
    Threshold,    # (M, 1)
    TILE_SIZE_M: ConstInt,
    TILE_SIZE_N: ConstInt,
    TILE_SIZE_V: ConstInt,
    REAL_V: ConstInt
):
    bid = ct.bid(0)
    
    # 1. Tile-based Indexing
    h_curr = ct.load(H_current, index=(bid, 0), shape=(TILE_SIZE_M, TILE_SIZE_N))
    h_next = ct.load(H_next, index=(bid, 0), shape=(TILE_SIZE_M, TILE_SIZE_N))
    mask = ct.load(Active_Mask, index=(bid, 0), shape=(TILE_SIZE_M, 1))
    steps = ct.load(Steps_Taken, index=(bid, 0), shape=(TILE_SIZE_M, 1))
    threshold = ct.load(Threshold, index=(bid, 0), shape=(TILE_SIZE_M, 1))
    logits = ct.load(Logits, index=(bid, 0), shape=(TILE_SIZE_M, TILE_SIZE_V), padding_mode=ct.PaddingMode.ZERO)
    
    # 2. Advanced Masking
    max_logit = ct.max(logits, axis=1, keepdims=True)
    is_real_row = max_logit > -1e10
    
    col_idx = ct.arange(TILE_SIZE_V, dtype=ct.int32)[None, :]
    is_real_col = col_idx < REAL_V
    
    logits_centered = logits - ct.where(is_real_row, max_logit, 0.0)
    exp_logits = ct.exp(logits_centered)
    sum_exp = ct.sum(ct.where(is_real_row & is_real_col, exp_logits, 0.0), axis=1, keepdims=True)
    
    # 3. Decision
    max_prob = ct.where(is_real_row & (sum_exp > 0), 1.0 / sum_exp, 0.0)
    was_active = mask > 0.5
    is_active_next = (max_prob < threshold) & was_active
    
    # 4. Update & Store
    ct.store(H_current, index=(bid, 0), tile=ct.where(was_active, h_next, h_curr))
    ct.store(Active_Mask, index=(bid, 0), tile=ct.astype(is_active_next, ct.float32))
    ct.store(Steps_Taken, index=(bid, 0), tile=steps + ct.astype(was_active, ct.int32))
