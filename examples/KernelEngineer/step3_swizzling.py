import cuda.tile as ct
import cupy as cp
import numpy as np

# Level 3: Swizzling
# Fix: Reorder tile processing for L2 Cache Locality

dev_id = cp.cuda.get_device_id()
NUM_SMS = cp.cuda.runtime.getDeviceProperties(dev_id)['multiProcessorCount']
M, N, K = 4096, 4096, 4096
TILE_SIZE = 128
GROUP_SIZE = 4

@ct.kernel
def kernel(A, B, C):
    bid = ct.bid(0)
    num_progs = ct.num_blocks(0)
    grid_n = N // TILE_SIZE
    total_tiles = (M // TILE_SIZE) * grid_n
    
    for t_id in range(bid, total_tiles, num_progs):
        # SWIZZLING LOGIC
        num_bid_in_group = GROUP_SIZE * grid_n
        group_id = t_id // num_bid_in_group
        first_bid_m = group_id * GROUP_SIZE
        
        bid_m = first_bid_m + (t_id % GROUP_SIZE)
        bid_n = (t_id % num_bid_in_group) // GROUP_SIZE
        
        if bid_m >= (M // TILE_SIZE) or bid_n >= grid_n: continue

        acc = ct.zeros((TILE_SIZE, TILE_SIZE), dtype=ct.float32)
        for k in range(K // TILE_SIZE):
            a = ct.load(A, (bid_m, k), (TILE_SIZE, TILE_SIZE))
            b = ct.load(B, (k, bid_n), (TILE_SIZE, TILE_SIZE))
            acc = ct.mma(ct.astype(a, ct.float16), ct.astype(b, ct.float16), acc)
        ct.store(C, (bid_m, bid_n), ct.astype(acc, C.dtype))

def main():
    print(f"=== Level 3: Swizzling ===")
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