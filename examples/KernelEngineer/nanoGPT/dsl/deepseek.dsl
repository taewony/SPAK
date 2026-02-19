# nanoGPT_snf.dsl
# GPT model forward pass in Semantic Normal Form (SNF)
# Designed for formal verification and cuTile implementation

param:
  # Model dimensions
  vocab_size : int                         # vocabulary size
  block_size : int                          # maximum context length (T)
  n_layer : int                              # number of transformer blocks
  n_head : int                                # number of attention heads per layer
  n_embd : int                                # embedding dimension (D)
  dropout : float                             # dropout probability
  bias : bool                                  # whether to use bias in linear layers

  # Derived dimensions
  D = n_embd
  H = n_head
  T = block_size
  head_dim = D // H                           # dimension per attention head

input:
  # Tokenized input sequence [batch_size, T]
  idx : Tensor([B, T], int64)

  # Optional past key/value caches for incremental decoding
  past_keys : Tensor([n_layer, B, H, T_past, head_dim], float32)?   # [L, B, H, T_past, D/H]
  past_values : Tensor([n_layer, B, H, T_past, head_dim], float32)?

parameter:
  # Token and position embeddings
  token_embedding : Tensor([vocab_size, D], float32)
  position_embedding : Tensor([T, D], float32)

  # Transformer block parameters (for each layer l in 0..n_layer-1)
  ln1_weight : Tensor([n_layer, D], float32)
  ln1_bias : Tensor([n_layer, D], float32) if bias else None

  # Attention projections: qkv combined or separate
  attn_q_weight : Tensor([n_layer, D, D], float32)
  attn_q_bias : Tensor([n_layer, D], float32) if bias else None
  attn_k_weight : Tensor([n_layer, D, D], float32)
  attn_k_bias : Tensor([n_layer, D], float32) if bias else None
  attn_v_weight : Tensor([n_layer, D, D], float32)
  attn_v_bias : Tensor([n_layer, D], float32) if bias else None
  attn_proj_weight : Tensor([n_layer, D, D], float32)
  attn_proj_bias : Tensor([n_layer, D], float32) if bias else None

  ln2_weight : Tensor([n_layer, D], float32)
  ln2_bias : Tensor([n_layer, D], float32) if bias else None

  # MLP (typically 4x expansion, GELU activation)
  mlp_fc_weight : Tensor([n_layer, D, 4*D], float32)
  mlp_fc_bias : Tensor([n_layer, 4*D], float32) if bias else None
  mlp_proj_weight : Tensor([n_layer, 4*D, D], float32)
  mlp_proj_bias : Tensor([n_layer, D], float32) if bias else None

  # Final layer norm and output head
  ln_f_weight : Tensor([D], float32)
  ln_f_bias : Tensor([D], float32) if bias else None
  lm_head_weight : Tensor([vocab_size, D], float32)  # often tied to token_embedding

output:
  # Logits for next token prediction [batch_size, T, vocab_size]
  logits : Tensor([B, T, vocab_size], float32)

  # Updated key/value caches
  new_keys : Tensor([n_layer, B, H, T_total, head_dim], float32)
  new_values : Tensor([n_layer, B, H, T_total, head_dim], float32)
    where T_total = T_past + T

# --- Core Computation ---

# 1. Token and position embeddings
let tok_emb = gather(token_embedding, idx, axis=0)                    # [B, T, D]
let pos_emb = slice(position_embedding, [0:T, :])                     # [T, D]
let x = tok_emb + pos_emb                                              # [B, T, D]
x = dropout(x, rate=dropout) if dropout > 0

# 2. Initialize KV cache arrays
let total_T = (past_keys.shape[3] if past_keys exists else 0) + T
let new_keys = full([n_layer, B, H, total_T, head_dim], 0.0, float32)
let new_values = full([n_layer, B, H, total_T, head_dim], 0.0, float32)

if past_keys exists:
    # Copy past caches
    new_keys = assign(new_keys, [:, :, :, :T_past, :], past_keys)
    new_values = assign(new_values, [:, :, :, :T_past, :], past_values)

# 3. Process each transformer block
for l in 0..n_layer-1:
    # --- Attention sub-block ---
    # Pre-normalization
    let x_norm = layer_norm(x, weight=ln1_weight[l], bias=ln1_bias[l])   # [B, T, D]

    # Project to Q, K, V (combined projection possible for efficiency)
    let q = linear(x_norm, weight=attn_q_weight[l], bias=attn_q_bias[l])  # [B, T, D]
    let k = linear(x_norm, weight=attn_k_weight[l], bias=attn_k_bias[l])  # [B, T, D]
    let v = linear(x_norm, weight=attn_v_weight[l], bias=attn_v_bias[l])  # [B, T, D]

    # Reshape for multi-head attention
    let q_head = reshape(q, [B, T, H, head_dim])                          # [B, T, H, D/H]
    let k_head = reshape(k, [B, T, H, head_dim])
    let v_head = reshape(v, [B, T, H, head_dim])

    # Combine with past KV if available
    let k_full = if past_keys exists:
        concat([past_keys[l], k_head], axis=1)                           # [B, T_past+T, H, D/H]
    else: k_head
    let v_full = if past_values exists:
        concat([past_values[l], v_head], axis=1)
    else: v_head

    # Update caches
    new_keys[l] = k_full
    new_values[l] = v_full

    # Causal self-attention (using Flash Attention pattern)
    # scale = 1.0 / sqrt(head_dim)
    let attn_output = flash_attention(
        Q=q_head,                        # [B, T, H, D/H]
        K=k_full,                        # [B, T_total, H, D/H]
        V=v_full,                         # [B, T_total, H, D/H]
        causal=True,
        scale=1.0 / sqrt(head_dim)
    )                                                                    # [B, T, H, D/H]

    # Merge heads and project
    let attn_merged = reshape(attn_output, [B, T, D])                    # [B, T, D]
    let attn_proj = linear(attn_merged,
                          weight=attn_proj_weight[l],
                          bias=attn_proj_bias[l])                         # [B, T, D]
    attn_proj = dropout(attn_proj, rate=dropout) if dropout > 0

    # Residual connection
    x = x + attn_proj                                                      # [B, T, D]

    # --- MLP sub-block ---
    let x_norm_mlp = layer_norm(x, weight=ln2_weight[l], bias=ln2_bias[l]) # [B, T, D]

    # MLP: typically linear -> GELU -> linear
    let mlp_hidden = linear(x_norm_mlp,
                           weight=mlp_fc_weight[l],
                           bias=mlp_fc_bias[l])                            # [B, T, 4*D]
    mlp_hidden = gelu(mlp_hidden)                                          # [B, T, 4*D]
    let mlp_out = linear(mlp_hidden,
                        weight=mlp_proj_weight[l],
                        bias=mlp_proj_bias[l])                             # [B, T, D]
    mlp_out = dropout(mlp_out, rate=dropout) if dropout > 0

    # Residual connection
    x = x + mlp_out                                                         # [B, T, D]

# 4. Final layer norm and output projection
let x_norm_final = layer_norm(x, weight=ln_f_weight, bias=ln_f_bias)      # [B, T, D]
let logits = linear(x_norm_final, weight=lm_head_weight, bias=None)       # [B, T, vocab_size]

# Note: Training loss (cross-entropy) would be computed separately