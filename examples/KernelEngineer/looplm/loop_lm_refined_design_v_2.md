# LoopLM Refined Engineering Design

Version: 2.0 (Implementation-Ready)
Status: Execution Blueprint
Target: cuTile-based LoopLM Prototype (nanoGPT-compatible core)

---

# 1. System Objective

LoopLM is a weight-tied recurrent Transformer that performs iterative latent refinement instead of static depth stacking.

Core Principle:

    h_{t+1} = h_t + F_θ( LN(h_t + LN(x0)) )

Where:
- F_θ is a shared Transformer block
- x0 is the original embedding anchor (persistent semantic injection)
- Recurrence occurs in time, not depth

Goal:
Enable adaptive computation depth and OOD reasoning while maintaining GPU efficiency via cuTile kernels.

---

# 2. Architectural Overview

## 2.1 Hierarchical Stack

Model SNF
  ↓
Operator SNF (matmul, layernorm, activation, attention)
  ↓
Tile-Kernel SNF (cuTile DSL)
  ↓
CUDA Kernel

LoopLM introduces temporal recurrence above Operator SNF.

---

# 3. Core Mathematical Specification

## 3.1 State Definition

Let:

- x0 ∈ ℝ[B,T,D]     (input embedding)
- h_t ∈ ℝ[B,T,D]   (latent state at step t)
- θ_shared         (shared block parameters)

Initialization:

    h_0 = x0

Update Rule:

    h_{t+1} = Mask_t ⊙ Block( h_t + LN(x0) )
              + (1 - Mask_t) ⊙ h_t

Where:
- Block = Residual(SelfAttention + MLP)
- Mask_t enables token-wise early halt

---

# 4. Module Decomposition (Implementation Units)

## 4.1 Embedding Module

Responsibilities:
- Token embedding
- Positional encoding (RoPE compatible)
- Produce x0

Output:
    x0 ∈ ℝ[B,T,D]

---

## 4.2 Shared Update Operator (Weight-Tied Transformer Block)

Block(x):

    a  = Attention( LN1(x) )
    x1 = x + a

    m  = MLP( LN2(x1) )
    x2 = x1 + m

Return x2

Constraints:
- Single parameter set θ_shared
- No layer stacking

---

## 4.3 Loop Executor (Core Engine)

Pseudo-code:

    h = x0
    active_mask = ones(B,T)

    for t in range(max_steps):
        h_input = h + LN(x0)
        h_next  = Block(h_input)

        logits  = LM_head(h_next)
        confidence = max(logits)

        done = confidence > threshold
        active_mask = active_mask & (~done)

        h = where(active_mask, h_next, h)

        if all(done): break

Outputs:
- final state h
- trajectory {h_t}
- step_count per token

---

# 5. cuTile Kernel Design

## 5.1 Kernel Responsibilities

Each iteration executes:
- LayerNorm
- Flash Attention (streaming softmax recurrence)
- MLP (matmul + activation)

Kernel signature:

    looplm_temporal_kernel(
        X_current,
        X0,
        W_attn,
        W_mlp,
        Mask_active,
        ...
    )

---

## 5.2 X0 Persistent Injection

- Load X0 once per block via TMA
- Keep in registers or shared memory
- Reuse each iteration

---

## 5.3 Masked Early Exit

Per-token halt:

    h = ct.where(mask[:,None], h_next, h)

Warp-level exit:

    if ct.all(done): break

---

# 6. Training System Design

LoopLM training differs from standard LM.

## 6.1 Multi-Step Supervision

Loss per step:

    L_total = Σ_t w_t * LM_loss(h_t)

Where weights encourage:
- Early convergence
- Stability across steps

---

## 6.2 Backpropagation Through Time (BPTT)

Gradients accumulate across recurrence:

    ∂L/∂h_t += (∂h_{t+1}/∂h_t)^T ∂L/∂h_{t+1}

Requirements:
- Save h_t per step
- No in-place overwrite during training

---

## 6.3 Regularization

Optional additions:

- Entropy regularization on halt gate
- Norm stabilization penalty
- Consistency loss across depths

---

# 7. Trace & Verification Schema

## 7.1 Kernel-Level Checks

- Softmax row sum ≈ 1
- No NaNs
- Norm stability

## 7.2 Temporal Stability Checks

- ||h_t|| bounded
- Convergence trend detection
- Step-depth correlation

## 7.3 Adaptive Compute Validation

Measure:

- average steps per difficulty bucket
- OOD generalization vs training depth

---

# 8. Experimental Protocol

## Phase 1 — Numerical Validation
- 1-step loop = standard GPT

## Phase 2 — Stability Validation
- Multi-step norm convergence

## Phase 3 — Algorithmic Tasks
- Multi-digit addition
- Parentheses matching
- Sorting trace
- Stack simulation

Evaluate:
- depth usage
- OOD length scaling

---

# 9. Engineering Milestones

1. Single-step weight-tied block equivalence
2. 2-step fixed loop training stable
3. Variable-step with halt gate
4. Adaptive depth emergence
5. OOD length generalization

---

# 10. Design Philosophy

LoopLM is not a deeper Transformer.

It is:

    A dynamical system performing iterative latent refinement.

Intelligence emerges from:
- Recurrence
- State persistence
- Adaptive halting

---

# 11. Minimal Implementation Order

1. Standard Transformer (weight tied, no loop)
2. Fixed 2-step loop
3. Multi-step BPTT
4. Halt gate integration
5. Kernel optimization via cuTile

---

End of Document

