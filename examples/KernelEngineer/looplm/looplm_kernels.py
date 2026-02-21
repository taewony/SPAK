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
    
    # 1. Load Data
    h_curr = ct.load(H_current, index=(bid, 0), shape=(TILE_SIZE_M, TILE_SIZE_N))
    h_next = ct.load(H_next, index=(bid, 0), shape=(TILE_SIZE_M, TILE_SIZE_N))
    mask = ct.load(Active_Mask, index=(bid, 0), shape=(TILE_SIZE_M, 1))
    steps = ct.load(Steps_Taken, index=(bid, 0), shape=(TILE_SIZE_M, 1))
    
    # FIX: Use a very small constant for padding to ensure exp(-1e20) = 0
    # This prevents the ZERO-exp(0)=1 bias pointed out by the user.
    logits = ct.load(Logits, index=(bid, 0), shape=(TILE_SIZE_M, TILE_SIZE_V), 
                     padding_mode=ct.PaddingMode.CONSTANT, constant_value=-1e20)
    
    # 2. Robust Max Probability Calculation
    max_logit = ct.max(logits, axis=1, keepdims=True)
    
    # NaN Guard: If max_logit is too small (padding row), it's not a real token.
    is_real_token = max_logit > -1e10
    # For calculation stability, we use a safe_max that is never -inf
    safe_max = ct.where(is_real_token, max_logit, 0.0)
    
    logits_centered = logits - safe_max
    exp_logits = ct.exp(logits_centered)
    
    # Sum only real logits
    sum_exp = ct.sum(ct.where(is_real_token, exp_logits, 1e20), axis=1, keepdims=True)
    
    # Correct max_prob: 1.0 / sum_exp (stable version of exp(max)/sum(exp))
    # If not a real token, max_prob is 0.0 (won't trigger halt prematurely)
    max_prob = ct.where(is_real_token, 1.0 / sum_exp, 0.0)
    
    # 3. Decision Logic
    was_active = mask > 0.5
    is_active_next = (max_prob < Threshold) & was_active
    
    # 4. State Update & Store
    h_updated = ct.where(was_active, h_next, h_curr)
    mask_updated = ct.astype(is_active_next, ct.float32)
    steps_updated = steps + ct.astype(was_active, ct.int32)
    
    ct.store(H_current, index=(bid, 0), tile=h_updated)
    ct.store(Active_Mask, index=(bid, 0), tile=mask_updated)
    ct.store(Steps_Taken, index=(bid, 0), tile=steps_updated)
