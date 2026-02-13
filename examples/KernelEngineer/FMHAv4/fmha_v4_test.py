import torch
import cuda.tile as ct
import math
import argparse
import time

# --- DSL-Guided Kernel Implementation ---
@ct.kernel(occupancy=2)
def fmha_v4_kernel(
    Q, K, V, Out,
    qk_scale_log2: float,
    TILE_D: ct.Constant[int],
    H: ct.Constant[int],
    TILE_M: ct.Constant[int],
    TILE_N: ct.Constant[int],
    QUERY_GROUP_SIZE: ct.Constant[int],
    K_LAT: ct.Constant[int],
    V_LAT: ct.Constant[int],
    EVEN_K: ct.Constant[bool],
    CAUSAL: ct.Constant[bool],
):
    bid_x, bid_y = ct.bid(0), ct.bid(1)
    batch_idx, head_idx = bid_y // H, bid_y % H
    off_kv_h = head_idx // QUERY_GROUP_SIZE
    
    m_i = ct.full((TILE_M, 1), -math.inf, dtype=ct.float32)
    l_i = ct.full((TILE_M, 1), 0.0, dtype=ct.float32)
    acc = ct.full((TILE_M, TILE_D), 0.0, dtype=ct.float32)

    offs_m = bid_x * TILE_M + ct.arange(TILE_M, dtype=ct.int32)[:, None]
    offs_n_tile = ct.arange(TILE_N, dtype=ct.int32)[None, :]

    q = ct.load(Q, index=(batch_idx, head_idx, bid_x, 0), shape=(1, 1, TILE_M, TILE_D)).reshape((TILE_M, TILE_D))

    k_seqlen = K.shape[2]
    
    if CAUSAL:
        m_end = (bid_x + 1) * TILE_M
        Tc = ct.cdiv(min(m_end, k_seqlen), TILE_N)
    else:
        Tc = ct.cdiv(k_seqlen, TILE_N)

    for j in range(0, Tc):
        k = ct.load(K, index=(batch_idx, off_kv_h, 0, j), shape=(1, 1, TILE_D, TILE_N), order=(0,1,3,2), latency=K_LAT).reshape((TILE_D, TILE_N))
        qk = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
        qk = ct.mma(q, k, qk)

        if CAUSAL or not EVEN_K:
            offs_n = j * TILE_N + offs_n_tile
            mask = ct.full((TILE_M, TILE_N), True, dtype=ct.bool_)
            if not EVEN_K: mask = mask & (offs_n < k_seqlen)
            if CAUSAL: mask = mask & (offs_m >= offs_n)
            qk = qk + ct.where(mask, 0.0, -math.inf)

        m_ij = max(m_i, ct.max(qk, axis=-1, keepdims=True) * qk_scale_log2)
        p = ct.exp2(qk * qk_scale_log2 - m_ij)
        l_ij = ct.sum(p, axis=-1, keepdims=True)
        alpha = ct.exp2(m_i - m_ij)
        
        l_i = l_i * alpha + l_ij
        acc = acc * alpha

        v = ct.load(V, index=(batch_idx, off_kv_h, j, 0), shape=(1, 1, TILE_N, TILE_D), latency=V_LAT).reshape((TILE_N, TILE_D))
        acc = ct.mma(p.astype(Q.dtype), v, acc)
        m_i = m_ij

    acc = ct.truediv(acc, l_i)
    ct.store(Out, index=(batch_idx, head_idx, bid_x, 0), tile=acc.reshape((1, 1, TILE_M, TILE_D)).astype(Out.dtype))

def benchmark(args):
    B, H, S, D = 8, 16, 2048, 128 # Increased size for stable measurement
    H_KV = H // 4 
    device_name = torch.cuda.get_device_name()
    
    q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H_KV, S, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H_KV, S, D, device='cuda', dtype=torch.float16)
    o = torch.zeros_like(q)
    
    scale = 1.0 / math.sqrt(D)
    qk_scale_log2 = scale * (1.0 / math.log(2))
    grid = (S // args.tile_m, B * H, 1)
    is_causal = args.causal == 1
    
    # --- Correctness Check ---
    # Broadcast K/V to match heads for PyTorch SDPA (GQA handling)
    k_ref = k.repeat_interleave(H // H_KV, dim=1)
    v_ref = v.repeat_interleave(H // H_KV, dim=1)
    
    with torch.no_grad():
        ref_out = torch.nn.functional.scaled_dot_product_attention(q, k_ref, v_ref, is_causal=is_causal, scale=scale)
    
    ct.launch(torch.cuda.current_stream(), grid, fmha_v4_kernel, 
             (q, k, v, o, qk_scale_log2, D, H, args.tile_m, args.tile_n, 4, args.klat, args.vlat, True, is_causal))
    torch.cuda.synchronize()
    
    max_diff = (o - ref_out).abs().max().item()
    passed = max_diff < 1e-2
    
    # --- Benchmarking ---
    iters = 100
    
    # cuTile
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        ct.launch(torch.cuda.current_stream(), grid, fmha_v4_kernel, 
                 (q, k, v, o, qk_scale_log2, D, H, args.tile_m, args.tile_n, 4, args.klat, args.vlat, True, is_causal))
    torch.cuda.synchronize()
    t1 = time.time()
    cutile_ms = (t1 - t0) * 1000 / iters
    
    # PyTorch
    torch.cuda.synchronize()
    t2 = time.time()
    for _ in range(iters):
        torch.nn.functional.scaled_dot_product_attention(q, k_ref, v_ref, is_causal=is_causal, scale=scale)
    torch.cuda.synchronize()
    t3 = time.time()
    torch_ms = (t3 - t2) * 1000 / iters
    
    # TFLOPS calculation (standard formula)
    ops = 4 * B * H * S * S * D
    cutile_tflops = ops / (cutile_ms * 1e-3) / 1e12
    torch_tflops = ops / (torch_ms * 1e-3) / 1e12
    
    print(f"DEVICE={device_name}")
    print(f"PASSED={passed} (MaxDiff={max_diff:.5f})")
    print(f"RESULT: TFLOPS={cutile_tflops:.2f}, TorchTFLOPS={torch_tflops:.2f}, Speedup={cutile_tflops/torch_tflops:.2f}x")
    print(f"PARAMS: TileM={args.tile_m}, TileN={args.tile_n}, KLat={args.klat}, VLat={args.vlat}, Causal={args.causal}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile_m", type=int, default=64)
    parser.add_argument("--tile_n", type=int, default=64)
    parser.add_argument("--klat", type=int, default=2)
    parser.add_argument("--vlat", type=int, default=4)
    parser.add_argument("--causal", type=int, default=1)
    args = parser.parse_args()
    benchmark(args)
