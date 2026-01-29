system AttentionOptimizer {

    kernel NaiveAttention {
        # Procedure: naive_attention_kernel
        # Input: Q[B, H, M, D], K[B, H, N, D], V[B, H, N, D]
        # Output: O[B, H, M, D]
        # Logic: 3-Stage Global Memory (QK -> Softmax -> PV)

        procedure naive_attention_kernel:
            # 1. QK Gemm (Write to Global)
            S = Q @ K.T / sqrt(D)
            
            # 2. Softmax (Read S, Write P)
            # Requires loading entire row N to find Max/Sum
            P = softmax(S, dim=-1)
            
            # 3. PV Gemm (Read P, Write O)
            O = P @ V
    }

    kernel FusedAttention {
        # Procedure: fused_attention_online_softmax
        # Input: Q, K, V (Same shapes)
        # Output: O (Same shape)
        # Optimization: Tiling + Online Softmax + Fusion

        procedure fused_attention_kernel:
            # 3D Grid: [Query_Tile_M, Batch*Head, 1]
            m_tile = blockIdx.x
            bh_idx = blockIdx.y
            
            # Init Registers for Online Softmax
            m_i = -inf  # Max
            l_i = 0.0   # Sum
            acc = 0.0   # Output Accumulator
            
            # Load Query Tile (SRAM)
            q_tile = load(Q[bh_idx, m_tile]) 
            
            # Loop over Key/Value Tiles (N dimension)
            for n_tile in 0..num_n_tiles-1:
                k_tile = load(K[bh_idx, n_tile])
                v_tile = load(V[bh_idx, n_tile])
                
                # 1. QK Gemm (Register)
                s_tile = q_tile @ k_tile.T
                
                # 2. Online Softmax Update
                m_new = max(m_i, max(s_tile))
                p_tile = exp(s_tile - m_new)
                l_new = sum(p_tile) + l_i * exp(m_i - m_new)
                
                # 3. Rescale Accumulator
                acc = acc * exp(m_i - m_new)
                
                # 4. PV Gemm (Register)
                acc += p_tile @ v_tile
                
                # Update State
                m_i = m_new
                l_i = l_new
            
            # Final Normalization & Store
            O[bh_idx, m_tile] = acc / l_i
    }
}
