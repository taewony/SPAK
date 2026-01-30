import cuda.tile as ct
import cupy as cp
import torch
import numpy as np

# ============================================================
# FMHA Step 2: Naive Kernel (Baseline)
# Strategy: 3 Separate Steps (QK -> Softmax -> PV) with Global Memory writes
# This simulates the high memory traffic of non-fused attention.
# ============================================================

B, H, M, N, D = 1, 4, 1024, 1024, 64
TILE_M = 64
TILE_N = 64

@ct.kernel
def naive_attention_kernel(Q, K, V, O, S_buf, P_buf):
    # We process one Tile of Q (M-dim) per block
    bid_m = ct.bid(0) 
    bid_bh = ct.bid(1) # Batch * Head
    
    # Indices
    b_idx = bid_bh // H
    h_idx = bid_bh % H
    
    # 1. QK GEMM (Write S to Global/Buffer)
    # Load Q Tile
    # Fix: Use reshape instead of slicing [0,0]
    q_tile = ct.load(Q, (b_idx, h_idx, bid_m, 0), (1, 1, TILE_M, D)).reshape((TILE_M, D))
    
    # Iterate over all N tiles to compute S row
    num_n_tiles = N // TILE_N
    
    # Note: A true naive kernel would compute ALL S then ALL P.
    # To avoid OOM in this test harness, we do it row-wise but FORCE memory roundtrips.
    
    # Pass 1: Compute Scores (Q @ K.T) -> Global Memory (S_buf)
    for j in range(num_n_tiles):
        # Fix: K is 4D
        k_tile = ct.load(K, (b_idx, h_idx, j, 0), (1, 1, TILE_N, D)).reshape((TILE_N, D))
        s_tile = ct.mma(q_tile, k_tile.T) # Q[M,D] @ K[N,D].T -> [M,N]
        # In Naive, we WRITE this to global memory
        # Fix: S_buf is 6D, use 6 indices. s_tile is 2D.
        # Unsqueeze s_tile to 6D to match S_buf rank
        ct.store(S_buf, (b_idx, h_idx, bid_m, j, 0, 0), s_tile[None, None, None, None])

    ct.commit() # Force memory sync (simulation)
    
    # Pass 2: Softmax (Read S -> Exp/Sum -> Write P)
    # Load entire row S for this tile (Naive Softmax requirement)
    # Simplified: We assume we can load it back tile-by-tile for max/sum calc
    m_max = ct.full((TILE_M, 1), -1e30, dtype=ct.float32)
    l_sum = ct.full((TILE_M, 1), 0.0, dtype=ct.float32)
    
    # 2a. Find Max
    for j in range(num_n_tiles):
        # Fix: S_buf is 6D
        s_tile = ct.load(S_buf, (b_idx, h_idx, bid_m, j, 0, 0), (1, 1, 1, 1, TILE_M, TILE_N)).reshape((TILE_M, TILE_N))
        m_max = ct.max(m_max, ct.max(s_tile, dim=1, keepdims=True))
        
    # 2b. Compute Exp & Sum
    for j in range(num_n_tiles):
        # Fix: S_buf is 6D
        s_tile = ct.load(S_buf, (b_idx, h_idx, bid_m, j, 0, 0), (1, 1, 1, 1, TILE_M, TILE_N)).reshape((TILE_M, TILE_N))
        p_tile = ct.exp(s_tile - m_max)
        l_sum = l_sum + ct.sum(p_tile, dim=1, keepdims=True)
        # Fix: P_buf is 6D
        ct.store(P_buf, (b_idx, h_idx, bid_m, j, 0, 0), p_tile[None, None, None, None])
        
    ct.commit()
        
    # Pass 3: PV GEMM (Read P -> Read V -> Write O)
    acc = ct.zeros((TILE_M, D), dtype=ct.float32)
    
    for j in range(num_n_tiles):
        # Fix: P_buf is 6D
        p_tile = ct.load(P_buf, (b_idx, h_idx, bid_m, j, 0, 0), (1, 1, 1, 1, TILE_M, TILE_N)).reshape((TILE_M, TILE_N))
        # Fix: V is 4D
        v_tile = ct.load(V, (b_idx, h_idx, j, 0), (1, 1, TILE_N, D)).reshape((TILE_N, D))
        
        # Normalize P here ( P_buf holds exp(S-m) )
        p_norm = p_tile / l_sum
        
        acc = ct.mma(p_norm, v_tile, acc)
        
    # Fix: O is 4D
    # For store to 4D with 2D tile, we might need to reshape tile or rely on index?
    # Trying with 4 indices and hoping store accepts 2D tile for remaining dims.
    # Note: If this fails, we might need to unsqueeze acc.
    ct.store(O, (b_idx, h_idx, bid_m, 0), acc[None, None])

def main():
    print(f"=== FMHA Step 2: Naive Kernel ({M}x{N}) ===")
    
    # Data
    d_Q = cp.random.randn(B, H, M, D).astype(cp.float16)
    d_K = cp.random.randn(B, H, N, D).astype(cp.float16)
    d_V = cp.random.randn(B, H, N, D).astype(cp.float16)
    d_O = cp.zeros((B, H, M, D), dtype=cp.float16)
    
    # Intermediate Buffers (The enemy of speed)
    d_S = cp.zeros((B, H, M // TILE_M, N // TILE_N, TILE_M, TILE_N), dtype=cp.float32) # Padded layout for tiled access
    d_P = cp.zeros_like(d_S)
    
    grid = (M // TILE_M, B * H, 1)
    
    # Warmup
    print("Warming up...")
    stream = cp.cuda.get_current_stream()
    for _ in range(3):
        ct.launch(stream, grid, naive_attention_kernel, (d_Q, d_K, d_V, d_O, d_S, d_P))
    stream.synchronize()
    
    # Measure
    print("Benchmarking...")
    start = cp.cuda.Event(); end = cp.cuda.Event()
    start.record()
    for _ in range(10):
        ct.launch(stream, grid, naive_attention_kernel, (d_Q, d_K, d_V, d_O, d_S, d_P))
    end.record()
    end.synchronize()
    
    ms = cp.cuda.get_elapsed_time(start, end) / 10.0
    # FLOPs: 2*M*N*D (QK) + 2*M*N*D (PV) = 4*M*N*D
    tflops = (4 * B * H * M * N * D) / (ms * 1e-3) / 1e12
    
    print("-" * 60)
    print(f"Time: {ms:.3f} ms | TFLOPS: {tflops:.2f}")
    print("-" * 60)
    
    # Correctness (using PyTorch as oracle)
    t_Q = torch.as_tensor(d_Q, device='cuda').float()
    t_K = torch.as_tensor(d_K, device='cuda').float()
    t_V = torch.as_tensor(d_V, device='cuda').float()
    
    ref_O = torch.nn.functional.scaled_dot_product_attention(t_Q, t_K, t_V, scale=1.0)
    
    res_O = torch.as_tensor(d_O, device='cuda').float()
    
    if torch.allclose(ref_O, res_O, atol=1e-1, rtol=1e-2):
        print("✅ Verification: Success!")
    else:
        diff = (ref_O - res_O).abs().max().item()
        print(f"❌ Verification: Failed (Max Diff: {diff:.4f})")

if __name__ == "__main__":
    main()