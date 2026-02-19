SNF_MODEL nanoGPT

# --------------------------------------------------
# 1. Tensor Domains
# --------------------------------------------------

Domain:
    B : batch
    T : sequence_position
    D : embedding_dim
    H : attention_heads
    L : layer_depth
    V : vocab

State:
    h[L+1] ∈ ℝ[B,T,D]     # layer states
    tok   ∈ ℕ[B,T]        # token ids

Parameters:
    θ_attn[L]
    θ_mlp[L]
    θ_norm[L]
    θ_embed
    θ_unembed

# --------------------------------------------------
# 2. Embedding
# --------------------------------------------------

Init:
    h[0] = Embed(tok, θ_embed)

# --------------------------------------------------
# 3. Attention Operator (reuse FMHA SNF)
# --------------------------------------------------

Operator Attention_l(x):

    q = Linear(x, θ_attn[l].Wq)
    k = Linear(x, θ_attn[l].Wk)
    v = Linear(x, θ_attn[l].Wv)

    p = CausalSoftmax(q @ k^T / √D)

    return p @ v

# --------------------------------------------------
# 4. FeedForward Operator
# --------------------------------------------------

Operator MLP_l(x):

    u = Linear(x, θ_mlp[l].W1)
    u = GELU(u)
    u = Linear(u, θ_mlp[l].W2)

    return u

# --------------------------------------------------
# 5. Layer Transition (Residual Dynamical System)
# --------------------------------------------------

Transition Layer_l:

    x  = h[l]

    a  = Attention_l(Norm(x, θ_norm[l].attn))
    x1 = x + a

    m  = MLP_l(Norm(x1, θ_norm[l].mlp))
    x2 = x1 + m

    h[l+1] = x2

# --------------------------------------------------
# 6. Depth Recurrence
# --------------------------------------------------

For l in [0 .. L-1]:
    apply Layer_l

# --------------------------------------------------
# 7. Output Projection
# --------------------------------------------------

logits = Linear(h[L], θ_unembed)

# --------------------------------------------------
# 8. Training Objective
# --------------------------------------------------

Loss:
    CrossEntropy(logits, tok shifted by 1)
