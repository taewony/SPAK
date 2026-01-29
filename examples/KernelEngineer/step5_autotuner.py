import cuda.tile as ct
import cupy as cp
import torch
import numpy as np
from dataclasses import dataclass
from contextlib import contextmanager

# ============================================================
# SPAK Auto-Tuned MatMul
# Merges 'cutile' Auto-Tuning with 'SPAK' Pipelining
# ============================================================

@dataclass
class Config:
    tile_m: int
    tile_n: int
    tile_k: int
    occupancy: int  # 1, 2, 4
    
    @property
    def num_ctas(self):
        return NUM_SMS * self.occupancy

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

# ============================================================
# 1. Kernel Factory (Closures for JIT Constant Folding)
# ============================================================
def make_pipelined_kernel(M, N, K, cfg: Config):
    """
    Returns a Kernel Function with constants (Tile Sizes) baked in.
    Implements SPAK's Double-Buffering Pipeline.
    """
    # 1. Capture Constants for JIT
    tile_m = cfg.tile_m
    tile_n = cfg.tile_n
    tile_k = cfg.tile_k
    
    # Grid Constants
    grid_m = (M + tile_m - 1) // tile_m
    grid_n = (N + tile_n - 1) // tile_n
    total_tiles = grid_m * grid_n
    num_k_tiles = (K + tile_k - 1) // tile_k
    
    GROUP_SIZE_M = 8

    def _kernel(A, B, C):
        bid = ct.bid(0)
        num_programs = ct.num_blocks(0)
        
        # Persistent Thread Block Loop
        for tile_id in range(bid, total_tiles, num_programs):
            
            # --- Swizzling ---
            num_bid_in_group = GROUP_SIZE_M * grid_n
            group_id = tile_id // num_bid_in_group
            first_bid_m = group_id * GROUP_SIZE_M
            group_size_m = min(grid_m - first_bid_m, GROUP_SIZE_M)
            
            bid_m = first_bid_m + (tile_id % group_size_m)
            bid_n = (tile_id % num_bid_in_group) // group_size_m
            
            # --- Pipelining Setup ---
            acc = ct.zeros((tile_m, tile_n), dtype=ct.float32)
            
            # Use TFloat32 if input is float32
            dtype_a = A.dtype
            
            # Prologue: Load first tile
            a_curr = ct.load(A, index=(bid_m, 0), shape=(tile_m, tile_k), padding_mode=ct.PaddingMode.ZERO)
            b_curr = ct.load(B, index=(0, bid_n), shape=(tile_k, tile_n), padding_mode=ct.PaddingMode.ZERO)
            
            # --- Main Loop (Double Buffered) ---
            for k_idx in range(num_k_tiles):
                # Identify tiles to compute in this iteration
                a_compute = a_curr
                b_compute = b_curr
                
                # Prefetch next tile (if not last iteration)
                if k_idx < num_k_tiles - 1:
                    a_curr = ct.load(A, index=(bid_m, k_idx + 1), shape=(tile_m, tile_k), padding_mode=ct.PaddingMode.ZERO)
                    b_curr = ct.load(B, index=(k_idx + 1, bid_n), shape=(tile_k, tile_n), padding_mode=ct.PaddingMode.ZERO)
                
                # Compute (overlaps with Load)
                # Note: cutile perf script casts to float16 here, we keep it generic or follow input
                acc = ct.mma(a_compute, b_compute, acc)
                
            # --- Epilogue ---
            ct.store(C, index=(bid_m, bid_n), tile=ct.astype(acc, C.dtype))
            
    return _kernel

# ============================================================
# 2. Benchmarking Engine
# ============================================================
def run_benchmark(M, N, K, configs, stream):
    results = []
    
    # Generate Data
    try:
        # Use Float16 for Tensor Cores (RTX 5070 optimization)
        d_A = cp.random.randn(M, K, dtype=cp.float32).astype(cp.float16)
        d_B = cp.random.randn(K, N, dtype=cp.float32).astype(cp.float16)
        d_C = cp.zeros((M, N), dtype=cp.float16)
        
        t_A = torch.as_tensor(d_A, device='cuda')
        t_B = torch.as_tensor(d_B, device='cuda')
    except cp.cuda.memory.OutOfMemoryError:
        return "OOM", []

    # Baseline: PyTorch
    for _ in range(5): torch.matmul(t_A, t_B)
    torch.cuda.synchronize()
    
    # Calculate Reference for Verification
    ref_C = torch.matmul(t_A, t_B)

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    
    start_evt.record()
    for _ in range(20): torch.matmul(t_A, t_B)
    end_evt.record()
    torch.cuda.synchronize()
    
    torch_ms = start_evt.elapsed_time(end_evt) / 20.0
    torch_tflops = (2.0 * M * N * K) / (torch_ms * 1e-3) / 1e12

    best_spak_tflops = 0
    best_spak_ms = 0 # Initialize ms
    best_cfg_str = ""

    # Sweep Configs
    for cfg in configs:
        kernel_source = make_pipelined_kernel(M, N, K, cfg)
        
        try:
            # Compile with timeout to skip bad configs quickly
            with compiler_timeout(5):
                kernel = ct.kernel(kernel_source, occupancy=cfg.occupancy)
        except Exception as e:
            # print(f"Skip {cfg}: {e}")
            continue

        # Launch
        grid = (cfg.num_ctas, 1, 1)
        args = (d_A, d_B, d_C)
        
        def run_kernel():
            ct.launch(stream, grid, kernel, args)
            
        stream.synchronize()
        run_kernel() # Single run for verification
        stream.synchronize()

        # Verification
        c_tensor = torch.as_tensor(d_C, device='cuda')
        if not torch.allclose(ref_C, c_tensor, atol=1e-1, rtol=1e-2):
            # print(f"Config {cfg} FAILED verification")
            continue

        for _ in range(4): run_kernel() # Finish Warmup (1+4=5)
        stream.synchronize()
        
        start_evt.record()
        for _ in range(20): run_kernel() # Measure
        end_evt.record()
        torch.cuda.synchronize()
        
        spak_ms = start_evt.elapsed_time(end_evt) / 20.0
        spak_tflops = (2.0 * M * N * K) / (spak_ms * 1e-3) / 1e12
        
        results.append((cfg, spak_ms, spak_tflops))
        
        if spak_tflops > best_spak_tflops:
            best_spak_tflops = spak_tflops
            best_spak_ms = spak_ms # Track best ms
            best_cfg_str = f"{cfg.tile_m}x{cfg.tile_n}x{cfg.tile_k} (Occ={cfg.occupancy})"

    del d_A, d_B, d_C, t_A, t_B
    cp.get_default_memory_pool().free_all_blocks()
    
    return (torch_tflops, best_spak_tflops, best_spak_ms, best_cfg_str, results) # Added spak_ms return

# ============================================================
# 3. Main
# ============================================================
def main():
    global NUM_SMS
    dev_id = cp.cuda.get_device_id()
    props = cp.cuda.runtime.getDeviceProperties(dev_id)
    NUM_SMS = props['multiProcessorCount']
    
    print(f">> SPAK Auto-Tuner on {props['name'].decode()} | SMs: {NUM_SMS}")
    print(">> Strategy: Double-Buffered Pipeline + Config Sweep")
    print("-" * 125)
    print(f"{'Size':<10} | {'PyTorch':<8} | {'SPAK Tuned':<10} | {'Time':<8} | {'Speedup':<8} | {'Best Config':<35}")
    print(f"{'(MxNxK)':<10} | {'(TFLOPS)':<8} | {'(TFLOPS)':<10} | {'(ms)':<8} | {'(%)':<8}     | {'(Tile_M x Tile_N x Tile_K)'}")
    print("-" * 125)

    # Search Space
    configs = [
        # Standard High Throughput
        Config(128, 128, 64, occupancy=1),
        Config(128, 128, 32, occupancy=1),
        # Balanced
        Config(128, 64, 64, occupancy=2),
        Config(128, 64, 32, occupancy=2),
        # High Occupancy (Latency Hiding)
        Config(64, 64, 64, occupancy=2),
        Config(64, 64, 64, occupancy=4),
        Config(64, 64, 32, occupancy=4),
    ]

    sizes = [2048, 4096, 8192] # Add 16384 if you have patience/RAM
    stream = cp.cuda.get_current_stream()

    for size in sizes:
        M = N = K = size
        ret = run_benchmark(M, N, K, configs, stream)
        
        if ret == "OOM":
            print(f"{size:<10} | OOM")
            continue
            
        torch_tf, spak_tf, spak_ms, best_cfg, _ = ret # Unpack ms
        speedup = (spak_tf - torch_tf) / torch_tf * 100
        sign = "+" if speedup > 0 else ""
        
        print(f"{size:<10} | {torch_tf:<8.2f} | {spak_tf:<10.2f} | {spak_ms:<8.3f} | {sign}{speedup:<7.1f}% | {best_cfg}")

    print("-" * 125)

if __name__ == "__main__":
    main()
