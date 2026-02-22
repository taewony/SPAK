# =============================================================================
# looplm_kernels.py – Semantic Norm / Pseudo Code Representation
# =============================================================================
#
# This file defines the CUDA kernel (using cuTile) that performs the halting
# update for LoopLM inference.
# =============================================================================

import math
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

# -----------------------------------------------------------------------------
# Kernel: looplm_halt_update_kernel
# -----------------------------------------------------------------------------
# This kernel is launched with a grid of (M_padded // TILE_SIZE_M) blocks.
# Each block processes a tile of TILE_SIZE_M rows (tokens) and the full
# TILE_SIZE_N (hidden) and TILE_SIZE_V (logits) columns.
#
# Inputs (all padded to tile dimensions):
#   H_current   – current hidden states (M_padded, TILE_N)
#   H_next      – next hidden states from transformer block (M_padded, TILE_N)
#   Logits      – logits for the step (M_padded, TILE_V)
#   Active_Mask – current active mask (M_padded, 1)
#   Steps_Taken – step counter (M_padded, 1)
#   Threshold   – halt confidence threshold
#   TILE_SIZE_M, TILE_SIZE_N, TILE_SIZE_V – compile‑time constants
#   REAL_V      – actual vocabulary size (to mask out padding columns)
#
# Output (in‑place updates):
#   H_current   – updated hidden states (frozen for halted tokens)
#   Active_Mask – updated active mask (0 = halted, 1 = still active)
#   Steps_Taken – incremented for tokens that were active this step
# -----------------------------------------------------------------------------
@ct.kernel
def looplm_halt_update_kernel(
    H_current,      # (M_padded, TILE_N)
    H_next,         # (M_padded, TILE_N)
    Logits,         # (M_padded, TILE_V)
    Active_Mask,    # (M_padded, 1)
    Steps_Taken,    # (M_padded, 1)
    Threshold: float,
    TILE_SIZE_M: ConstInt,
    TILE_SIZE_N: ConstInt,
    TILE_SIZE_V: ConstInt,
    REAL_V: int
):
    # Get block index (each block processes one horizontal strip)
    bid = ct.bid(0)

    # Load tiles for this block
    h_curr = ct.load(H_current, index=(bid, 0), shape=(TILE_SIZE_M, TILE_SIZE_N))
    h_next = ct.load(H_next,     index=(bid, 0), shape=(TILE_SIZE_M, TILE_SIZE_N))
    mask   = ct.load(Active_Mask, index=(bid, 0), shape=(TILE_SIZE_M, 1))
    steps  = ct.load(Steps_Taken, index=(bid, 0), shape=(TILE_SIZE_M, 1))
    logits = ct.load(Logits,      index=(bid, 0), shape=(TILE_SIZE_M, TILE_SIZE_V),
                     padding_mode=ct.PaddingMode.ZERO)   # pad with zero outside the allocated region

    # ---- Step 1: Compute max probability (over the REAL vocabulary) ----
    # Only the first REAL_V columns correspond to actual tokens.
    # We set logits for columns >= REAL_V to -inf so they do not contribute.
    # However, because we used ZERO padding outside the allocated region,
    # we must explicitly mask the padded columns.

    # Create a mask for valid vocabulary columns
    col_idx = ct.arange(0, TILE_SIZE_V, 1)   # column indices
    valid_mask = col_idx < REAL_V             # 1 for real vocab, 0 for padding

    # Apply mask: set logits of padding columns to -inf
    logits_masked = ct.where(valid_mask, logits, -float('inf'))

    # Compute max over columns (stable softmax)
    max_logit = ct.max(logits_masked, axis=1, keepdims=True)

    # Shift and exponentiate
    logits_centered = logits_masked - max_logit
    exp_logits = ct.exp(logits_centered)

    # Mask again to zero out contributions from padding columns
    exp_logits = ct.where(valid_mask, exp_logits, 0.0)

    sum_exp = ct.sum(exp_logits, axis=1, keepdims=True)

    # Softmax (only needed for max probability)
    softmax = exp_logits / sum_exp
    max_prob = ct.max(softmax, axis=1, keepdims=True)

    # ---- Step 2: Determine which tokens remain active ----
    # A token was active before this step if mask > 0.5.
    was_active = mask > 0.5

    # It stays active if max_prob < Threshold (and it was active).
    stays_active = (max_prob < Threshold) & was_active

    # ---- Step 3: Update state, mask, and step counter ----
    # Hidden state: if was active, take h_next; otherwise keep h_curr.
    h_updated = ct.where(was_active, h_next, h_curr)

    # Active mask: cast stays_active to float32
    mask_updated = ct.astype(stays_active, ct.float32)

    # Steps taken: increment only if was_active.
    steps_updated = steps + ct.astype(was_active, ct.int32)

    # ---- Step 4: Store results back to global memory ----
    ct.store(H_current,   index=(bid, 0), tile=h_updated)
    ct.store(Active_Mask, index=(bid, 0), tile=mask_updated)
    ct.store(Steps_Taken, index=(bid, 0), tile=steps_updated)
