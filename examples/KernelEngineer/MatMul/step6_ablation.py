import cuda.tile as ct
import cupy as cp
import torch
import numpy as np
from dataclasses import dataclass
from contextlib import contextmanager

# ============================================================
# Step 6: Ablation Study (Pipelining vs No-Pipelining)
# Goal: Isolate the performance gain of Double Buffering
# ============================================================

@dataclass
class Config:
    tile_m: int
    tile_n: int
    tile_k: int
    occupancy: int

@contextmanager
def compiler_timeout(timeout_sec: int):
    try:
        from cuda.tile._cext import default_tile_context
        old = default_tile_context.config.compiler_timeout_sec
        default_tile_context.config.compiler_timeout_sec = timeout_sec
        yield
        default_tile_context.config.compiler_timeout_sec = old
    except (ImportError, AttributeError):
        yield

# Factory: Can generate BOTH Pipelined and Simple kernels for the same Config
def make_kernel_source(M, N, K, cfg, use_pipelining=True):
    tile_m = cfg.tile_m
    tile_n = cfg.tile_n
    tile_k = cfg.tile_k
    
    grid_m = (M + tile_m - 1) // tile_m
    grid_n = (N + tile_n - 1) // tile_n
    total_tiles = grid_m * grid_n
    num_k_tiles = (K + tile_k - 1) // tile_k
    
    GROUP_SIZE_M = 8

    def _kernel(A, B, C):
        bid = ct.bid(0)
        num_programs = ct.num_blocks(0)
        
        for tile_id in range(bid, total_tiles, num_programs):
            # --- Common Swizzling Logic ---
            num_bid_in_group = GROUP_SIZE_M * grid_n
            group_id = tile_id // num_bid_in_group
            first_bid_m = group_id * GROUP_SIZE_M
            group_size_m = min(grid_m - first_bid_m, GROUP_SIZE_M)
            
            bid_m = first_bid_m + (tile_id % group_size_m)
            bid_n = (tile_id % num_bid_in_group) // group_size_m
            
            acc = ct.zeros((tile_m, tile_n), dtype=ct.float32)
            
            if use_pipelining:
                # === OPTION A: Double Buffered Pipeline ===
                a_curr = ct.load(A, (bid_m, 0), (tile_m, tile_k))
                b_curr = ct.load(B, (0, bid_n), (tile_k, tile_n))
                
                for k_idx in range(num_k_tiles):
                    a_compute = a_curr
                    b_compute = b_curr
                    
                    if k_idx < num_k_tiles - 1:
                        a_curr = ct.load(A, (bid_m, k_idx + 1), (tile_m, tile_k))
                        b_curr = ct.load(B, (k_idx + 1, bid_n), (tile_k, tile_n))
                    
                    acc = ct.mma(a_compute, b_compute, acc)

            else:
                # === OPTION B: Simple Sequential Loop ===
                for k_idx in range(num_k_tiles):
                    # Load and Wait
                    a = ct.load(A, (bid_m, k_idx), (tile_m, tile_k))
                    b = ct.load(B, (k_idx, bid_n), (tile_k, tile_n))
                    # Compute
                    acc = ct.mma(a, b, acc)

            ct.store(C, (bid_m, bid_n), ct.astype(acc, C.dtype))
            
    return _kernel

def benchmark_config(M, N, K, cfg, use_pipelining, stream):
    label = "Pipelined" if use_pipelining else "No-Pipe"
    
    # Generate Kernel
    source = make_kernel_source(M, N, K, cfg, use_pipelining)
    try:
        with compiler_timeout(5):
            kernel = ct.kernel(source, occupancy=cfg.occupancy)
    except Exception as e:
        return 0.0

    # Data
    d_A = cp.random.randn(M, K).astype(cp.float16)
    d_B = cp.random.randn(K, N).astype(cp.float16)
    d_C = cp.zeros((M, N), dtype=cp.float16)
    
    # SM Check
    dev_id = cp.cuda.get_device_id()
    num_sms = cp.cuda.runtime.getDeviceProperties(dev_id)['multiProcessorCount']
    grid = (num_sms * cfg.occupancy, 1, 1)

    # Measure
    args = (d_A, d_B, d_C)
    
    # Warmup
    for _ in range(5): ct.launch(stream, grid, kernel, args)
    stream.synchronize()
    
    start = cp.cuda.Event(); end = cp.cuda.Event(); start.record()
    for _ in range(20): ct.launch(stream, grid, kernel, args)
    end.record(); end.synchronize()
    
    ms = cp.cuda.get_elapsed_time(start, end) / 20.0
    tflops = (2 * M * N * K) / (ms * 1e-3) / 1e12
    return tflops

def main():
    dev_id = cp.cuda.get_device_id()
    props = cp.cuda.runtime.getDeviceProperties(dev_id)
    print(f"=== Step 6: Ablation Study on {props['name'].decode()} ===")
    
    M, N, K = 4096, 4096, 4096
    stream = cp.cuda.get_current_stream()
    
    # The 'Winner' Config from Step 5 (RTX 5070 optimal)
    # Typically 64x64x64 with Occupancy 2
    best_cfg = Config(64, 64, 64, occupancy=2)
    
    print(f"Target Config: Tile={best_cfg.tile_m}x{best_cfg.tile_n}x{best_cfg.tile_k}, Occ={best_cfg.occupancy}")
    print("-" * 60)
    print(f"{'Mode':<20} | {'TFLOPS':<10} | {'Speedup':<10}")
    print("-" * 60)
    
    # Run Baseline (No Pipe)
    tf_baseline = benchmark_config(M, N, K, best_cfg, use_pipelining=False, stream=stream)
    print(f"{'Simple Loop':<20} | {tf_baseline:<10.2f} | 1.00x")
    
    # Run Optimized (Pipe)
    tf_pipe = benchmark_config(M, N, K, best_cfg, use_pipelining=True, stream=stream)
    speedup = tf_pipe / tf_baseline if tf_baseline > 0 else 0
    print(f"{'Double Buffered':<20} | {tf_pipe:<10.2f} | {speedup:.2f}x")
    
    print("-" * 60)
    gain = (tf_pipe - tf_baseline)
    if gain > 5.0:
        print(f"✅ Conclusion: Pipelining contributes significantly (+{gain:.1f} TFLOPS).")
    else:
        print(f"ℹ️ Conclusion: Impact is minor. Kernel might be Compute-Bound.")

    # DSL Trace Emission
    import json
    trace = {
        "type": "Performance",
        "step_name": "Level 6: Ablation Study",
        "tflops": tf_pipe,
        "speedup": speedup
    }
    # Also emit Analysis trace
    analysis = {
        "type": "Analysis",
        "bottleneck": "Pipeline_Stall" if gain > 5.0 else "DRAM_BW"
    }
    print(f"__SPAK_TRACE__{json.dumps(trace)}")
    print(f"__SPAK_TRACE__{json.dumps(analysis)}")

if __name__ == "__main__": main()
