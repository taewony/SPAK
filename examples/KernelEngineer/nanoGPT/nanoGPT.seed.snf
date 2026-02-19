SNF_MODEL LM_SEED

# ==================================================
# 0. Domain (only existence, no size)
# ==================================================

Domain:
    Batch
    Time
    Channel
    Depth
    Vocab

# ==================================================
# 1. State Variables
# ==================================================

State:
    token[Batch,Time]          # discrete input
    h[Depth+1, Batch,Time,Channel]   # latent trajectory
    logits[Batch,Time,Vocab]

Parameters:
    θ_embed
    θ_block
    θ_readout

# ==================================================
# 2. Uninterpreted Primitive Operators
# (semantics intentionally unknown)
# ==================================================

Operator Embed(token, θ_embed) -> latent
Operator Block(latent, θ_block) -> latent
Operator Readout(latent, θ_readout) -> logits
Operator NextTokenLoss(logits, token) -> scalar

# ==================================================
# 3. Latent Initialization
# ==================================================

Init:
    h[0] = Embed(token, θ_embed)

# ==================================================
# 4. Depth Recurrence
# ==================================================

For l in Depth:
    h[l+1] = Block(h[l], θ_block)

# ==================================================
# 5. Output Projection
# ==================================================

logits = Readout(h[Depth], θ_readout)

# ==================================================
# 6. Training Objective
# ==================================================

Loss:
    NextTokenLoss(logits, token)

# ==================================================
# 7. Learning Dynamics (minimal existence only)
# ==================================================

Update:
    θ ← Learn(θ, Loss)
