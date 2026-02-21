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
    TILE_SIZE_V: ConstInt
):
    bid = ct.bid(0)
    # 2D LAW: index must use pure tile index (bid) when shape is provided.
    # FIX: Removed manual multiply (* TILE_SIZE_M)
    h_curr = ct.load(H_current, index=(bid, 0), shape=(TILE_SIZE_M, TILE_SIZE_N))
    h_next = ct.load(H_next, index=(bid, 0), shape=(TILE_SIZE_M, TILE_SIZE_N))
    mask = ct.load(Active_Mask, index=(bid, 0), shape=(TILE_SIZE_M, 1))
    steps = ct.load(Steps_Taken, index=(bid, 0), shape=(TILE_SIZE_M, 1))
    
    logits = ct.load(Logits, index=(bid, 0), shape=(TILE_SIZE_M, TILE_SIZE_V), padding_mode=ct.PaddingMode.ZERO)
    
    # Mathematical Max Probability (Proven correct in RCA v3)
    max_logit = ct.max(logits, axis=1, keepdims=True)
    exp_logits = ct.exp(logits - max_logit)
    sum_exp = ct.sum(exp_logits, axis=1, keepdims=True)
    max_prob = 1.0 / sum_exp
    
    was_active = mask > 0.5
    is_active_next = (max_prob < Threshold) & was_active
    
    h_updated = ct.where(was_active, h_next, h_curr)
    mask_updated = ct.astype(is_active_next, ct.float32)
    steps_updated = steps + ct.astype(was_active, ct.int32)
    
    ct.store(H_current, index=(bid, 0), tile=h_updated)
    ct.store(Active_Mask, index=(bid, 0), tile=mask_updated)
    ct.store(Steps_Taken, index=(bid, 0), tile=steps_updated)
