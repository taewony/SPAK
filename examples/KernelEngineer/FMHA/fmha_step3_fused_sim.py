import torch
import math

# ============================================================ 
# FMHA Step 3: Fused Kernel Simulation (Python/PyTorch)
# Goal: Verify the Tiling + Online Softmax logic bit-exactness
#       without requiring the specific GPU/cuTile environment.
# ============================================================ 

def fused_attention_sim(Q, K, V, tile_m=128, tile_n=128, qk_scale=None, causal=False):
    """
    Simulates the structure of the Fused Attention Kernel.
    Iterates over Output Tiles (M) and Inner Input Tiles (N).
    """
    Batch, Heads, SeqLen_Q, D_k = Q.shape
    _, _, SeqLen_KV, D_v = V.shape
    
    if qk_scale is None:
        qk_scale = 1.0 / math.sqrt(D_k)

    # Output Tensor
    O = torch.zeros((Batch, Heads, SeqLen_Q, D_v), dtype=Q.dtype, device=Q.device)
    
    # Grid Loops (simulating GPU blocks)
    # Loop 1: Batch * Heads
    for b in range(Batch):
        for h in range(Heads):
            
            # Loop 2: Query Tiles (M dimension) - equivalent to blockIdx.x
            for m_start in range(0, SeqLen_Q, tile_m):
                m_end = min(m_start + tile_m, SeqLen_Q)
                m_size = m_end - m_start
                
                # Load Q Tile (SRAM simulation)
                # Shape: [m_size, D]
                q_tile = Q[b, h, m_start:m_end, :]
                
                # Initialize Accumulators (Registers)
                # m_i: Max per row (init to -inf)
                m_i = torch.full((m_size, 1), -float('inf'), device=Q.device, dtype=torch.float32)
                # l_i: Sum per row (init to 0)
                l_i = torch.full((m_size, 1), 0.0, device=Q.device, dtype=torch.float32)
                # acc: Weighted Sum (init to 0)
                acc = torch.zeros((m_size, D_v), device=Q.device, dtype=torch.float32)
                
                # Loop 3: Key/Value Tiles (N dimension) - The "Streaming" Loop
                for n_start in range(0, SeqLen_KV, tile_n):
                    n_end = min(n_start + tile_n, SeqLen_KV)
                    n_size = n_end - n_start
                    
                    # Load K, V Tiles (SRAM simulation)
                    k_tile = K[b, h, n_start:n_end, :] # [n_size, D]
                    v_tile = V[b, h, n_start:n_end, :] # [n_size, D]
                    
                    # 1. QK Gemm
                    # [m_size, D] @ [D, n_size] -> [m_size, n_size]
                    s_tile = torch.matmul(q_tile, k_tile.transpose(-2, -1))
                    
                    # Apply Causal Masking
                    if causal:
                        # Global indices
                        row_idx = torch.arange(m_start, m_end, device=Q.device)[:, None]
                        col_idx = torch.arange(n_start, n_end, device=Q.device)[None, :]
                        mask = row_idx >= col_idx
                        s_tile = torch.where(mask, s_tile, torch.tensor(-float('inf'), device=Q.device))

                    # 2. Online Softmax Update
                    
                    # Local Max in this tile
                    # [m_size, 1]
                    m_ij = torch.max(s_tile, dim=-1, keepdim=True).values * qk_scale
                    
                    # Local P (unnormalized exp)
                    # Note: s_tile is not yet scaled in the logic above?
                    # The reference code scales qk BEFORE max?
                    # "m_ij = max(m_i, ct.max(qk, ...) * scale)"
                    # "qk = qk * scale - m_ij"
                    # Let's match that exactly.
                    
                    s_tile_scaled = s_tile * qk_scale
                    m_current_max = torch.max(s_tile_scaled, dim=-1, keepdim=True).values
                    m_current_max = torch.where(m_current_max > -float('inf'), m_current_max, torch.tensor(-float('inf'), device=Q.device))

                    
                    # Update Global Max
                    m_new = torch.maximum(m_i, m_current_max)
                    
                    # Rescaling factors
                    # alpha = exp(m_old - m_new)
                    alpha = torch.exp(m_i - m_new)
                    # beta = exp(m_current - m_new)
                    beta = torch.exp(m_current_max - m_new)
                    
                    # P_tile calculation
                    # p = exp(s * scale - m_new)
                    #   = exp(s * scale - m_current) * exp(m_current - m_new)
                    #   = exp(s * scale - m_current) * beta
                    p_tile = torch.exp(s_tile_scaled - m_new)
                    
                    # Update Sum
                    # l_new = l_old * alpha + sum(p_tile)
                    l_current = torch.sum(p_tile, dim=-1, keepdim=True)
                    l_new = l_i * alpha + l_current
                    
                    # Update Accumulator (O)
                    # acc_new = acc_old * alpha + p_tile @ v_tile
                    # Note: in standard derivation, p_tile is already scaled by correct m_new
                    w_v = torch.matmul(p_tile.to(Q.dtype), v_tile)
                    acc = acc * alpha + w_v
                    
                    # Commit State
                    m_i = m_new
                    l_i = l_new
                    
                # Final Normalization
                # O = acc / l_i
                o_tile = acc / l_i
                
                # Store
                O[b, h, m_start:m_end, :] = o_tile.to(Q.dtype)

    return O

def main():
    print("=== FMHA Step 3: Python Logic Simulation ===")
    
    # 1. Setup
    torch.manual_seed(42)
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
        
    print(f"Device: {device}")
    
    B, H, M, N, D = 1, 4, 1024, 1024, 64
    Q = torch.randn(B, H, M, D, device=device, dtype=torch.float32)
    K = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
    V = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
    
    print(f"Shape: B={B}, H={H}, M={M}, N={N}, D={D}")
    
    # 2. Reference (Torch SDPA)
    print("Running Reference (Torch SDPA)...")
    ref_O = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=False)
    
    # 3. Simulation (Fused Logic)
    print("Running Simulation (Fused Logic)...")
    sim_O = fused_attention_sim(Q, K, V, tile_m=128, tile_n=128, causal=False)
    
    # 4. Verification
    print("Verifying...")
    
    # Check for NaNs
    if torch.isnan(sim_O).any():
        print("❌ Simulation produced NaNs!")
        return

    diff = (ref_O - sim_O).abs().max()
    print(f"Max Diff: {diff:.2e}")
    
    if torch.allclose(ref_O, sim_O, atol=1e-4, rtol=1e-4):
        print("✅ Logic Verification: Success!")
    else:
        print("❌ Logic Verification: Failed!")
        
    # Causal Test
    print("\n--- Causal Masking Test ---")
    ref_O_causal = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True)
    sim_O_causal = fused_attention_sim(Q, K, V, tile_m=128, tile_n=128, causal=True)
    
    diff_c = (ref_O_causal - sim_O_causal).abs().max()
    print(f"Max Diff (Causal): {diff_c:.2e}")
    
    if torch.allclose(ref_O_causal, sim_O_causal, atol=1e-4, rtol=1e-4):
         print("✅ Causal Logic Verification: Success!")
    else:
         print("❌ Causal Logic Verification: Failed!")

if __name__ == "__main__":
    main()
