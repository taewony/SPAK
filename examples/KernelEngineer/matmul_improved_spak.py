import cuda.tile as ct
import cupy as cp
import numpy as np
import torch

# ============================================================
# Optimized MatMul by SPAK (Semantic Programmable Agent Kernel)
# Based on NVIDIA Reference and Pipelining Heuristics
# ============================================================

dev_id = cp.cuda.get_device_id()
props = cp.cuda.runtime.getDeviceProperties(dev_id)
NUM_SMS = props['multiProcessorCount']
NUM_CTAS = NUM_SMS * 2 

M_SIZE = N_SIZE = K_SIZE = 4096
TILE_SIZE = 128

NUM_BID_M = M_SIZE // TILE_SIZE
NUM_BID_N = N_SIZE // TILE_SIZE
NUM_K_TILES = K_SIZE // TILE_SIZE

# --- Helper: Swizzle Logic from NVIDIA Reference ---
def swizzle_2d(M, N, tm, tn, GROUP_SIZE_M, bid):
    num_bid_m = ct.cdiv(M, tm)
    num_bid_n = ct.cdiv(N, tn)
    num_bid_in_group = GROUP_SIZE_M * num_bid_n
    group_id = bid // num_bid_in_group
    first_bid_m = group_id * GROUP_SIZE_M
    actual_group_size_m = min(num_bid_m - first_bid_m, GROUP_SIZE_M)
    
    bid_m = first_bid_m + (bid % actual_group_size_m)
    bid_n = (bid % num_bid_in_group) // actual_group_size_m
    return bid_m, bid_n

@ct.kernel
def matmul_kernel_spak_optimized(A, B, C, group_size_m):
    # Added group_size_m param to match original signature for compatibility if needed
    bid = ct.bid(0)
    num_blocks = ct.num_blocks(0)
    total_tiles = NUM_BID_M * NUM_BID_N
    
    # Override group size with NVIDIA optimal constant if passed as 0 or ignored
    # But let's use the param logic for benchmarking
    GROUP_SIZE_M = group_size_m if group_size_m > 0 else 8
    
    for current_bid in range(bid, total_tiles, num_blocks):
        # 1. Swizzling for Cache Locality
        bid_m, bid_n = swizzle_2d(M_SIZE, N_SIZE, TILE_SIZE, TILE_SIZE, GROUP_SIZE_M, current_bid)
        
        acc = ct.zeros((TILE_SIZE, TILE_SIZE), dtype=ct.float32)
        dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype
        
        # 2. Manual Pipelining (Double Buffering)
        # Prologue
        a_curr = ct.load(A, index=(bid_m, 0), shape=(TILE_SIZE, TILE_SIZE)).astype(dtype)
        b_curr = ct.load(B, index=(0, bid_n), shape=(TILE_SIZE, TILE_SIZE)).astype(dtype)
        
        for k in range(NUM_K_TILES):
            # Prefetch next tile
            if k < NUM_K_TILES - 1:
                a_next = ct.load(A, index=(bid_m, k+1), shape=(TILE_SIZE, TILE_SIZE)).astype(dtype)
                b_next = ct.load(B, index=(k+1, bid_n), shape=(TILE_SIZE, TILE_SIZE)).astype(dtype)
            
            # Compute current (overlaps with prefetch)
            acc = ct.mma(a_curr, b_curr, acc=acc)
            
            # Shift
            if k < NUM_K_TILES - 1:
                a_curr = a_next
                b_curr = b_next

        ct.store(C, index=(bid_m, bid_n), tile=ct.astype(acc, C.dtype))

# ============================================================
# Benchmarking Harness (Restored)
# ============================================================
def benchmark_cutile(label, d_A, d_B, d_C, group_size):
    stream = cp.cuda.get_current_stream()
    N_WARMUP = 5
    N_ITER = 20
    
    # 1. Warmup
    for _ in range(N_WARMUP):
        ct.launch(
            stream, (NUM_CTAS, 1, 1), 
            matmul_kernel_spak_optimized, 
            (d_A, d_B, d_C, group_size)
        )
    stream.synchronize()
    
    # 2. Measure
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()
    
    start_event.record()
    for _ in range(N_ITER):
        ct.launch(
            stream, (NUM_CTAS, 1, 1), 
            matmul_kernel_spak_optimized, 
            (d_A, d_B, d_C, group_size)
        )
    end_event.record()
    end_event.synchronize()
    
    avg_ms = cp.cuda.get_elapsed_time(start_event, end_event) / N_ITER
    print(f"{label:<25} : {avg_ms:.3f} ms")
    return avg_ms

def benchmark_pytorch(A_t, B_t):
    N_WARMUP = 5
    N_ITER = 20
    # Warmup
    for _ in range(N_WARMUP): _ = torch.matmul(A_t, B_t)
    torch.cuda.synchronize()
    
    # Measure
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(N_ITER): _ = torch.matmul(A_t, B_t)
    end.record()
    torch.cuda.synchronize()
    
    avg_ms = start.elapsed_time(end) / N_ITER
    print(f"{'PyTorch (cuBLAS)':<25} : {avg_ms:.3f} ms")
    return avg_ms

def main():
    print("=" * 60)
    print(f"=== SPAK Optimized Benchmark ({M_SIZE}x{N_SIZE}x{K_SIZE}) ===")
    print(f"SM Count: {NUM_SMS}, Active CTAs: {NUM_CTAS}")
    print("-" * 60)

    # Data Prep
    np.random.seed(0)
    # Using float16 for Tensor Core performance
    h_A = np.random.randn(M_SIZE, K_SIZE).astype(np.float16)
    h_B = np.random.randn(K_SIZE, N_SIZE).astype(np.float16)
    h_C = np.zeros((M_SIZE, N_SIZE), dtype=np.float16)

    d_A = cp.asarray(h_A)
    d_B = cp.asarray(h_B)
    d_C = cp.asarray(h_C)
    
    A_t = torch.from_numpy(h_A).cuda()
    B_t = torch.from_numpy(h_B).cuda()

    # --- Run Benchmarks ---
    t_torch = benchmark_pytorch(A_t, B_t)
    t_spak  = benchmark_cutile("SPAK (Swizzle+Pipeline)", d_A, d_B, d_C, group_size=8)

    # --- Verification ---
    print("-" * 60)
    print("Verifying correctness...")
    C_ref = torch.matmul(A_t, B_t).cpu().numpy()
    C_cutile = cp.asnumpy(d_C)
    
    # Relaxed tolerance for large FP16 accumulation
    if np.allclose(C_cutile, C_ref, atol=2e-1, rtol=2e-2):
        print("✅ Verification: Success!")
    else:
        diff = np.abs(C_cutile - C_ref)
        print(f"❌ Verification: Failed! Max error: {diff.max()}")

    # --- Final Report ---
    print("=" * 60)
    print(f"{'Method':<25} | {'Time (ms)':<10} | {'Rel. Speed':<10}")
    print("-" * 60)
    print(f"{'PyTorch (cuBLAS)':<25} | {t_torch:.3f} ms  | 1.00x")
    print(f"{'SPAK Optimized':<25}   | {t_spak:.3f} ms   | {t_torch/t_spak:.2f}x")
    print("=" * 60)

if __name__ == "__main__":
    main()
