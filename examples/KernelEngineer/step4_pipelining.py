import cuda.tile as ct
import cupy as cp
import numpy as np

# Level 4: Pipelining (Double Buffering)
# Fix: Overlap Memory Load with Computation
# NOTE: We use TILE_SIZE=64 here because we know 128 caused regression in previous attempts.
# This represents "Applying Pipelining Correctly".

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
    
    grid = (NUM_SMS * 2, 1, 1) 
    
    # Warmup
    for _ in range(5): ct.launch(cp.cuda.get_current_stream(), grid, kernel, (d_A, d_B, d_C))
    cp.cuda.Device().synchronize()

    # Measure
    start = cp.cuda.Event(); end = cp.cuda.Event(); start.record()
    for _ in range(20): ct.launch(cp.cuda.get_current_stream(), grid, kernel, (d_A, d_B, d_C))
    end.record(); end.synchronize()
    
    ms = cp.cuda.get_elapsed_time(start, end)/20
    tflops = (2 * M * N * K) / (ms * 1e-3) / 1e12

    print(f"Time: {ms:.3f} ms | TFLOPS: {tflops:.2f}")

if __name__ == "__main__": main()