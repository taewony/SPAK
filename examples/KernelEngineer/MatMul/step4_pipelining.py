import cuda.tile as ct
import cupy as cp
import torch
import numpy as np

# Level 4: Pipelining (Double Buffering)
# Fix: Overlap Memory Load with Computation
# NOTE: We use TILE_SIZE=64 here because we know 128 caused regression in previous attempts.

dev_id = cp.cuda.get_device_id()
NUM_SMS = cp.cuda.runtime.getDeviceProperties(dev_id)['multiProcessorCount']
M, N, K = 4096, 4096, 4096
TILE_SIZE = 64 # Tuned manually for Level 4 demo

@ct.kernel
def kernel(A, B, C):
    bid = ct.bid(0)
    num_progs = ct.num_blocks(0)
    grid_n = N // TILE_SIZE
    total_tiles = (M // TILE_SIZE) * grid_n
    num_k = K // TILE_SIZE

    for t_id in range(bid, total_tiles, num_progs):
        # Simple Swizzling
        bid_m = t_id // grid_n 
        bid_n = t_id % grid_n
        
        acc = ct.zeros((TILE_SIZE, TILE_SIZE), dtype=ct.float32)
        
        # PROLOGUE: Load first tile
        a_curr = ct.load(A, (bid_m, 0), (TILE_SIZE, TILE_SIZE))
        b_curr = ct.load(B, (0, bid_n), (TILE_SIZE, TILE_SIZE))
        
        for k in range(num_k):
            # PIPELINE: Identify what to compute
            a_compute = a_curr
            b_compute = b_curr
            
            # PIPELINE: Prefetch Next (if not last)
            if k < num_k - 1:
                a_curr = ct.load(A, (bid_m, k+1), (TILE_SIZE, TILE_SIZE))
                b_curr = ct.load(B, (k+1, bid_n), (TILE_SIZE, TILE_SIZE))
            
            # COMPUTE: Overlaps with the Load above
            # Optimization: Inputs are already float16, direct MMA
            acc = ct.mma(a_compute, b_compute, acc)
            
        ct.store(C, (bid_m, bid_n), ct.astype(acc, C.dtype))

def main():
    print(f"=== Level 4: Pipelining (Double Buffering) ===")
    
    d_A = cp.random.randn(M, K).astype(cp.float16)
    d_B = cp.random.randn(K, N).astype(cp.float16)
    d_C = cp.zeros((M, N), dtype=cp.float16)
    
    t_A = torch.as_tensor(d_A, device='cuda')
    t_B = torch.as_tensor(d_B, device='cuda')

    # Benchmark PyTorch
    for _ in range(5): torch.matmul(t_A, t_B)
    torch.cuda.synchronize()
    
    start_pt = torch.cuda.Event(enable_timing=True)
    end_pt = torch.cuda.Event(enable_timing=True)
    start_pt.record()
    for _ in range(20): torch.matmul(t_A, t_B)
    end_pt.record()
    torch.cuda.synchronize()
    pt_ms = start_pt.elapsed_time(end_pt) / 20.0

    # Benchmark Kernel
    grid = (NUM_SMS * 2, 1, 1) 
    
    for _ in range(5): ct.launch(cp.cuda.get_current_stream(), grid, kernel, (d_A, d_B, d_C))
    cp.cuda.Device().synchronize()

    start_k = cp.cuda.Event(); end_k = cp.cuda.Event(); start_k.record()
    for _ in range(20): ct.launch(cp.cuda.get_current_stream(), grid, kernel, (d_A, d_B, d_C))
    end_k.record(); end_k.synchronize()
    
    k_ms = cp.cuda.get_elapsed_time(start_k, end_k)/20
    k_tflops = (2 * M * N * K) / (k_ms * 1e-3) / 1e12

    # Correctness
    print("Verifying...", end=" ")
    ref_C = torch.matmul(t_A, t_B)
    res_C = torch.as_tensor(d_C, device='cuda')
    if torch.allclose(ref_C, res_C, atol=1e-1, rtol=1e-2):
        print("[OK] Correct")
    else:
        diff = (ref_C - res_C).abs().max().item()
        print(f"[FAIL] Failed (Max Diff: {diff:.4f})")

    # Report
    print("-" * 60)
    print(f"{'Method':<20} | {'Time (ms)':<10} | {'TFLOPS':<8} | {'Speedup':<8}")
    print("-" * 60)
    print(f"{'PyTorch':<20} | {pt_ms:<10.3f} | {'-':<8} | 1.00x")
    print(f"{'Pipelining':<20} | {k_ms:<10.3f} | {k_tflops:<8.2f} | {pt_ms/k_ms:.2f}x")
    print("-" * 60)
    
    print(f"Time: {k_ms:.3f} ms | TFLOPS: {k_tflops:.2f}")

    # DSL Trace Emission
    import json
    trace = {
        "type": "Performance",
        "step_name": "Level 4: Pipelining",
        "tflops": k_tflops,
        "speedup": pt_ms / k_ms
    }
    print(f"__SPAK_TRACE__{json.dumps(trace)}")

if __name__ == "__main__": main()
