import cuda.tile as ct
import cupy as cp
import numpy as np
import torch

# ============================================================
# 1. Problem size
# ============================================================
M_SIZE = 4096 # Updated to match benchmark standard
N_SIZE = 4096
K_SIZE = 4096

TILE_SIZE = 128   # tile size

GRID_M = M_SIZE // TILE_SIZE
GRID_N = N_SIZE // TILE_SIZE
GRID_K = K_SIZE // TILE_SIZE


# ============================================================
# 2. cuTile kernel
# ============================================================
@ct.kernel
def matmul_tile_kernel(A, B, C):
    # Block (tile) indices
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    # FP32 accumulator
    acc = ct.zeros((TILE_SIZE, TILE_SIZE), dtype=ct.float32)

    for k in range(GRID_K):
        # Load A tile
        tile_A = ct.load(
            A,
            index=(bid_m, k),
            shape=(TILE_SIZE, TILE_SIZE)
        )

        # Load B tile
        tile_B = ct.load(
            B,
            index=(k, bid_n),
            shape=(TILE_SIZE, TILE_SIZE)
        )

        # Convert for compute
        tile_A = ct.astype(tile_A, ct.float16)
        tile_B = ct.astype(tile_B, ct.float16)

        # Compute (lowered internally to MMA-safe form)
        acc += tile_A @ tile_B

    # Store result
    ct.store(
        C,
        index=(bid_m, bid_n),
        tile=ct.astype(acc, C.dtype)
    )


# ============================================================
# 3. Main
# ============================================================
def main():
    print(f"=== cuTile Baseline Benchmark ({M_SIZE}x{N_SIZE}x{K_SIZE}) ===")

    # --------------------------------------------------------
    # Host data
    # --------------------------------------------------------
    np.random.seed(0)
    h_A = np.random.randn(M_SIZE, K_SIZE).astype(np.float16)
    h_B = np.random.randn(K_SIZE, N_SIZE).astype(np.float16)
    h_C = np.zeros((M_SIZE, N_SIZE), dtype=np.float16)

    d_A = cp.asarray(h_A)
    d_B = cp.asarray(h_B)
    d_C = cp.asarray(h_C)
    
    A_torch = torch.from_numpy(h_A).cuda()
    B_torch = torch.from_numpy(h_B).cuda()

    # --------------------------------------------------------
    # Benchmarking
    # --------------------------------------------------------
    stream = cp.cuda.get_current_stream()
    
    # Warmup
    for _ in range(5):
        ct.launch((GRID_M, GRID_N, 1), matmul_tile_kernel, (d_A, d_B, d_C))
    stream.synchronize()
    
    # Measure
    start_evt = cp.cuda.Event(); end_evt = cp.cuda.Event()
    start_evt.record()
    for _ in range(20):
        ct.launch((GRID_M, GRID_N, 1), matmul_tile_kernel, (d_A, d_B, d_C))
    end_evt.record()
    end_evt.synchronize()
    
    ms = cp.cuda.get_elapsed_time(start_evt, end_evt) / 20.0
    tflops = (2 * M_SIZE * N_SIZE * K_SIZE) / (ms * 1e-3) / 1e12

    # --------------------------------------------------------
    # Verification
    # --------------------------------------------------------
    print("Verifying with PyTorch...")
    with torch.no_grad():
        C_ref = torch.matmul(A_torch, B_torch)
    C_ref_np = C_ref.cpu().numpy()
    result_cutile = cp.asnumpy(d_C)

    if np.allclose(result_cutile, C_ref_np, atol=2e-1, rtol=2e-2):
        print("✅ Verification: Success!")
    else:
        print("❌ Verification: Failed!")

    # --------------------------------------------------------
    # Standard Output for Reporter
    # --------------------------------------------------------
    print("-" * 60)
    print(f"Time: {ms:.3f} ms | TFLOPS: {tflops:.2f}")
    print("-" * 60)

if __name__ == "__main__":
    main()