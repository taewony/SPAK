import cuda.tile as ct
import cupy as cp
import torch
import numpy as np

# Level 2: Optimized Occupancy
# Fix: Launch enough CTAs to saturate the GPU

dev_id = cp.cuda.get_device_id()
NUM_SMS = cp.cuda.runtime.getDeviceProperties(dev_id)['multiProcessorCount']
M, N, K = 4096, 4096, 4096
TILE_SIZE = 128

@ct.kernel
def kernel(A, B, C):
    bid = ct.bid(0)
    num_progs = ct.num_blocks(0)
    total_tiles = (M // TILE_SIZE) * (N // TILE_SIZE)
    
    for t_id in range(bid, total_tiles, num_progs):
        bid_m = t_id // (N // TILE_SIZE)
        bid_n = t_id % (N // TILE_SIZE)
        
        acc = ct.zeros((TILE_SIZE, TILE_SIZE), dtype=ct.float32)
        for k in range(K // TILE_SIZE):
            a = ct.load(A, (bid_m, k), (TILE_SIZE, TILE_SIZE))
            b = ct.load(B, (k, bid_n), (TILE_SIZE, TILE_SIZE))
            acc = ct.mma(ct.astype(a, ct.float16), ct.astype(b, ct.float16), acc)
        ct.store(C, (bid_m, bid_n), ct.astype(acc, C.dtype))

def main():
    print(f"=== Level 2: Optimized Occupancy ===")
    d_A = cp.random.randn(M, K).astype(cp.float16)
    d_B = cp.random.randn(K, N).astype(cp.float16)
    d_C = cp.zeros((M, N), dtype=cp.float16)
    
    # FIX: Dynamic Grid Size
    grid = (NUM_SMS * 2, 1, 1) 
    print(f"Launching {grid[0]} CTAs on {NUM_SMS} SMs")
    
    start = cp.cuda.Event(); end = cp.cuda.Event(); start.record()
    for _ in range(20): ct.launch(cp.cuda.get_current_stream(), grid, kernel, (d_A, d_B, d_C))
    end.record(); end.synchronize()
    print(f"Time: {cp.cuda.get_elapsed_time(start, end)/20:.3f} ms")

if __name__ == "__main__": main()
