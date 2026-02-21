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
    Threshold: float,
    TILE_SIZE_M: ConstInt,
    TILE_SIZE_N: ConstInt,
    TILE_SIZE_V: ConstInt,
    REAL_V: ConstInt # Actual vocab size for column masking
):
    bid = ct.bid(0)
    
    # 1. Tile-based Indexing (Based on NVIDIA Doc)
    # i=bid, tm=TILE_SIZE_M -> Actual rows = bid * TILE_SIZE_M + x
    h_curr = ct.load(H_current, index=(bid, 0), shape=(TILE_SIZE_M, TILE_SIZE_N))
    h_next = ct.load(H_next, index=(bid, 0), shape=(TILE_SIZE_M, TILE_SIZE_N))
    mask = ct.load(Active_Mask, index=(bid, 0), shape=(TILE_SIZE_M, 1))
    steps = ct.load(Steps_Taken, index=(bid, 0), shape=(TILE_SIZE_M, 1))
    
    # Logits load with ZERO padding (standard support)
    logits = ct.load(Logits, index=(bid, 0), shape=(TILE_SIZE_M, TILE_SIZE_V), padding_mode=ct.PaddingMode.ZERO)
    
    # 2. Advanced Masking (Row & Column)
    max_logit = ct.max(logits, axis=1, keepdims=True)
    is_real_row = max_logit > -1e10 # Mask out padding tokens
    
    # Column Mask: Identify valid vocab indices within the tile
    # Use arange to get column indices and compare with REAL_V
    col_idx = ct.arange(TILE_SIZE_V, dtype=ct.int32)[None, :]
    is_real_col = col_idx < REAL_V
    
    # Subtract max for stability
    safe_max = ct.where(is_real_row, max_logit, 0.0)
    logits_centered = logits - safe_max
    
    # exp(logits) only for real rows AND real columns
    # This perfectly solves the ZERO-exp(0)=1 bias.
    exp_logits = ct.exp(logits_centered)
    safe_exp = ct.where(is_real_row & is_real_col, exp_logits, 0.0)
    
    sum_exp = ct.sum(safe_exp, axis=1, keepdims=True)
    
    # 3. Max Probability Decision
    # Correct formula: max_prob = exp(max-max) / sum(exp) = 1.0 / sum_exp
    max_prob = ct.where(is_real_row & (sum_exp > 0), 1.0 / sum_exp, 0.0)
    
    was_active = mask > 0.5
    is_active_next = (max_prob < Threshold) & was_active
    
    # 4. State Persistence & Store
    h_updated = ct.where(was_active, h_next, h_curr)
    mask_updated = ct.astype(is_active_next, ct.float32)
    steps_updated = steps + ct.astype(was_active, ct.int32)
    
    ct.store(H_current, index=(bid, 0), tile=h_updated)
    ct.store(Active_Mask, index=(bid, 0), tile=mask_updated)
    ct.store(Steps_Taken, index=(bid, 0), tile=steps_updated)
