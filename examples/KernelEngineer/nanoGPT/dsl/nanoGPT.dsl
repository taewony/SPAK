system "nanoGPT" {
    version: "2.0";
    description: "Consolidated Semantic Normal Form (SNF) for nanoGPT/GPT-2";
    backend: "cuTile/TileGym";

    # --------------------------------------------------
    # 1. Domains & Dimensions
    # --------------------------------------------------
    Domain:
        B : batch_size
        T : sequence_length (block_size)
        D : embedding_dim (n_embd)
        H : attention_heads (n_head)
        L : layer_depth (n_layer)
        V : vocab_size
        h_d : head_dim (D // H)

    # --------------------------------------------------
    # 2. State & Parameters
    # --------------------------------------------------
    State:
        h[L+1] ∈ ℝ[B,T,D]                   # Hidden states per layer
        kv_cache[L] ∈ (ℝ[B,H,T_past,h_d], ℝ[B,H,T_past,h_d])?

    Parameters:
        θ_embed   ∈ ℝ[V, D]                 # Token embedding (wte)
        θ_pos     ∈ ℝ[T, D]                 # Position embedding (wpe)
        θ_norm_f  ∈ {γ: ℝ[D], β: ℝ[D]}      # Final LayerNorm
        
        # Layer-specific parameters (l ∈ 0..L-1)
        θ_layer[L]:
            norm1 : {γ: ℝ[D], β: ℝ[D]}
            attn  : {Wqkv: ℝ[D, 3*D], proj: ℝ[D, D]}
            norm2 : {γ: ℝ[D], β: ℝ[D]}
            mlp   : {fc: ℝ[D, 4*D], proj: ℝ[4*D, D]}

    # --------------------------------------------------
    # 3. Low-level Operators (TileGym Mapping)
    # --------------------------------------------------
    Operator LayerNorm(x, θ):
        invariant: "y = (x - μ) / σ * θ.γ + θ.β"
        return cuTile.layernorm(x, θ.γ, θ.β)

    Operator FlashAttention(q, k, v, mask=Causal):
        reference: "kernels/fmha_blackwell.dsl"
        return cuTile.flash_attn(q, k, v, causal=True, scale=1/√h_d)

    Operator MLP(x, θ):
        u = cuTile.linear(x, θ.fc)
        u = cuTile.gelu(u, approx="tanh")
        return cuTile.linear(u, θ.proj)

    # --------------------------------------------------
    # 4. Layer Transition (Residual Dynamical System)
    # --------------------------------------------------
    Transition Block(x, θ_l):
        # Attention Path
        x_norm1 = LayerNorm(x, θ_l.norm1)
        q, k, v = cuTile.split(cuTile.linear(x_norm1, θ_l.attn.Wqkv), 3)
        
        # Reshape for multi-head [B, T, D] -> [B, H, T, h_d]
        q_h, k_h, v_h = map(λ t: reshape(t, [B, H, T, h_d]), [q, k, v])
        
        attn_out = FlashAttention(q_h, k_h, v_h)
        attn_proj = cuTile.linear(reshape(attn_out, [B, T, D]), θ_l.attn.proj)
        x1 = x + attn_proj                  # Residual 1

        # MLP Path
        x_norm2 = LayerNorm(x1, θ_l.norm2)
        mlp_out = MLP(x_norm2, θ_l.mlp)
        x2 = x1 + mlp_out                   # Residual 2
        
        return x2

    # --------------------------------------------------
    # 5. Forward Pass (Execution Trace)
    # --------------------------------------------------
    Trace Forward:
        input: tok ∈ ℕ[B, T]
        
        # Embedding
        tok_emb = gather(θ_embed, tok)
        pos_emb = slice(θ_pos, [0:T, :])
        h[0] = tok_emb + pos_emb
        
        # Depth Recurrence
        for l in [0 .. L-1]:
            h[l+1] = Block(h[l], θ_layer[l])
            
        # Output Projection
        h_final = LayerNorm(h[L], θ_norm_f)
        logits = cuTile.linear(h_final, θ_embed.T) # Weight Tying
        
        return logits

    # --------------------------------------------------
    # 6. Formal Invariants & Rules
    # --------------------------------------------------
    Invariant Weight_Tying:
        assert: "lm_head.weight == θ_embed.weight"
    
    Invariant Dimension_Preservation:
        assert: "forall l: shape(h[l]) == [B, T, D]"

    Rule Initialization:
        "Linear/Embedding ~ Normal(0, 0.02)"
        "Residual Projections scale by 1/√(2*L)"
}
