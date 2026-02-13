import torch
import cuda.tile as ct
import math
import argparse
import time

# --- DSL-Guided Kernel Implementation ---
@ct.kernel(occupancy=2)
def fmha_v4_kernel(
    Q, K, V, Out,
    qk_scale: float,
    TILE_D: ct.Constant[int],
    H: ct.Constant[int],
    TILE_M: ct.Constant[int],
    TILE_N: ct.Constant[int],
    QUERY_GROUP_SIZE: ct.Constant[int],
    K_LAT: ct.Constant[int],
    V_LAT: ct.Constant[int],
    EVEN_K: ct.Constant[bool],
):
    bid_x, bid_y = ct.bid(0), ct.bid(1)
    batch_idx, head_idx = bid_y // H, bid_y % H
    off_kv_h = head_idx // QUERY_GROUP_SIZE
    
    qk_scale = qk_scale * (1.0 / math.log(2)) # Corrected for exp2

    # Accumulators
    m_i = ct.full((TILE_M, 1), -math.inf, dtype=ct.float32)
    l_i = ct.full((TILE_M, 1), 0.0, dtype=ct.float32)
    acc = ct.full((TILE_M, TILE_D), 0.0, dtype=ct.float32)

    # Load Q
    q = ct.load(Q, index=(batch_idx, head_idx, bid_x, 0), shape=(1, 1, TILE_M, TILE_D)).reshape((TILE_M, TILE_D))

    k_seqlen = K.shape[2]
    Tc = ct.cdiv(k_seqlen, TILE_N)

    for j in range(0, Tc):
        # TMA Optimized Load for K
        k = ct.load(K, index=(batch_idx, off_kv_h, 0, j), shape=(1, 1, TILE_D, TILE_N), order=(0,1,3,2), latency=K_LAT).reshape((TILE_D, TILE_N))
        
        qk = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
        qk = ct.mma(q, k, qk)

        # Dynamic Masking (from DSL memory_robustness)
        if not EVEN_K:
            offs_n = j * TILE_N + ct.arange(TILE_N)[None, :]
            mask = ct.where(offs_n < k_seqlen, 0.0, -math.inf)
            qk += mask

        m_ij = max(m_i, ct.max(qk, axis=-1, keepdims=True) * qk_scale)
        p = ct.exp2(qk * qk_scale - m_ij)
        l_ij = ct.sum(p, axis=-1, keepdims=True)
        alpha = ct.exp2(m_i - m_ij)
        
        l_i = l_i * alpha + l_ij
        acc = acc * alpha

        # TMA Optimized Load for V
        v = ct.load(V, index=(batch_idx, off_kv_h, j, 0), shape=(1, 1, TILE_N, TILE_D), latency=V_LAT).reshape((TILE_N, TILE_D))
        acc = ct.mma(p.astype(Q.dtype), v, acc)
        m_i = m_ij

    acc = ct.truediv(acc, l_i)
    ct.store(Out, index=(batch_idx, head_idx, bid_x, 0), tile=acc.reshape((1, 1, TILE_M, TILE_D)).astype(Out.dtype))

def benchmark(args):
    B, H, S, D = 2, 32, 2048, 128
    H_KV = H // 4 # GQA factor of 4
    q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H_KV, S, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H_KV, S, D, device='cuda', dtype=torch.float16)
    o = torch.zeros_like(q)
    
    scale = 1.0 / math.sqrt(D)
    grid = (S // args.tile_m, B * H, 1)
    
    # Warmup
    for _ in range(10):
        ct.launch(torch.cuda.current_stream(), grid, fmha_v4_kernel, 
                 (q, k, v, o, scale, D, H, args.tile_m, args.tile_n, 4, args.klat, args.vlat, True))
    
    torch.cuda.synchronize()
    start = time.time()
    iters = 100
    for _ in range(iters):
        ct.launch(torch.cuda.current_stream(), grid, fmha_v4_kernel, 
                 (q, k, v, o, scale, D, H, args.tile_m, args.tile_n, 4, args.klat, args.vlat, True))
    torch.cuda.synchronize()
    end = time.time()
    
    avg_ms = (end - start) * 1000 / iters
    tflops = (2 * B * H * S * S * D) / (avg_ms * 1e-3) / 1e12
    print(f"RESULT: TFLOPS={tflops:.2f}, TileM={args.tile_m}, TileN={args.tile_n}, KLat={args.klat}, VLat={args.vlat}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile_m", type=int, default=64)
    parser.add_argument("--tile_n", type=int, default=64)
    parser.add_argument("--klat", type=int, default=2)
    parser.add_argument("--vlat", type=int, default=4)
    args = parser.parse_args()
    benchmark(args)
