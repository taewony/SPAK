SNF_MODEL nanoGPT refines LM_SEED

# ==================================================
# 0. Domain Refinement (Concrete Hyperparameters)
# ==================================================

Knowledge "GPT2_Small_Config":
    Batch       : B
    Time        : T (<= 1024)
    Channel     : C (n_embd = 768)
    Depth       : L (n_layer = 12)
    Vocab       : V (50257)
    Head        : H (n_head = 12)
    HeadDim     : D_h (C / H = 64)

# ==================================================
# 1. State Variables Refinement (Memory Layout)
# ==================================================

State:
    token[B, T]                : "int32"
    h[L+1, B, T, C]            : "bfloat16" (Residual Stream Trajectory)
    logits[B, T, V]            : "float32"

Parameters (Learnable):
    θ_wte[V, C]                : Token Embeddings
    θ_wpe[T, C]                : Positional Embeddings
    θ_blocks[L]                : Stack of Transformer Weights
    θ_ln_f[C]                  : Final LayerNorm Weights

# ==================================================
# 2. Operator Refinement (TileGym Primitives)
# ==================================================

# 2.1 Component Definitions
Module LayerNorm(x, θ) -> y:
    y = (x - mean) / std * θ.weight + θ.bias

Module CausalSelfAttention(x, θ) -> y:
    # "The Heart of the Transformer"
    q, k, v = Linear(x).split()
    att = Softmax(q @ k.T / sqrt(D_h) + Mask_Causal)
    y = Linear(att @ v)

Module MLP(x, θ) -> y:
    # "Channel-mixing"
    h = GELU(Linear(x, 4*C))
    y = Linear(h, C)

# 2.2 The Transformer Block Operator
Operator TransformerBlock(h_in, θ) -> h_out:
    # Pre-Norm Architecture (Stabilizes Training)
    # 1. Self-Attention Path
    n1 = LayerNorm(h_in, θ.ln1)
    h_mid = h_in + CausalSelfAttention(n1, θ.attn)
    
    # 2. Feed-Forward Path
    n2 = LayerNorm(h_mid, θ.ln2)
    h_out = h_mid + MLP(n2, θ.mlp)

# ==================================================
# 3. Latent Initialization (Embedding Sum)
# ==================================================

Init:
    # token embedding + positional embedding
    tok_emb = Gather(θ_wte, token)
    pos_emb = Gather(θ_wpe, Range(T))
    h[0] = tok_emb + pos_emb [Dropout if training]

# ==================================================
# 4. Depth Recurrence (The Residual Stream)
# ==================================================

Invariant "Residual_Connection":
    # The gradient highway exists across all layers
    h[l+1] = h[l] + SubModule(h[l])

Dynamics:
    For l in Range(L):
        h[l+1] = TransformerBlock(h[l], θ_blocks[l])

# ==================================================
# 5. Output Projection (Specific to GPT-2)
# ==================================================

Projection:
    # Final LayerNorm before projection (Distinct feature of GPT-2)
    h_final = LayerNorm(h[L], θ_ln_f)
    
    # Weight Tying Constraint: Readout uses WTE weights
    logits = Linear(h_final, weight=θ_wte)

# ==================================================
# 6. Training Objective & Constraints
# ==================================================

Loss:
    CrossEntropy(logits.view(-1, V), token.view(-1))

Constraint "NanoGPT_Specifics":
    1. WeightTying: "Readout head shares weights with Token Embedding"
    2. Bias: "False for Linears/Attn, True for LayerNorm"
    3. Init_Scale: "Residual projections scaled by 1/sqrt(2*L)"

# ==================================================
# 7. Update Rule (Optimization)
# ==================================================

Update:
    Optimizer: "AdamW(β1=0.9, β2=0.95, weight_decay=0.1)"
    Scheduler: "CosineDecay(warmup_iters=2000)"