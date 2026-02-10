import numpy as np
import torch
import torch.nn.functional as F

# ============================================================
# FMHA Step 1: Mathematical Invariant Check
# Goal: Prove Online Softmax (Stateful) == Standard Softmax (Stateless)
# ============================================================

def standard_attention(Q, K, V):
    """
    Standard "Safe" Softmax (3 passes over memory)
    1. Find Max (m)
    2. Compute Exp & Sum (l)
    3. Div & Matmul
    """
    scale = 1.0 / np.sqrt(Q.shape[-1])
    
    # 1. QK^T
    S = np.matmul(Q, K.transpose(0, 1, 3, 2)) * scale
    
    # 2. Softmax (Safe)
    m = np.max(S, axis=-1, keepdims=True)
    exp_S = np.exp(S - m)
    l = np.sum(exp_S, axis=-1, keepdims=True)
    P = exp_S / l
    
    # 3. PV
    O = np.matmul(P, V)
    return O

def online_softmax_attention(Q, K, V, tile_size=128):
    """
    Online Softmax (1 pass over memory using Tiling)
    Simulates the logic we will implement in the CUDA kernel.
    """
    B, H, M, D = Q.shape
    _, _, N, _ = K.shape
    scale = 1.0 / np.sqrt(D)
    
    O = np.zeros_like(Q)
    
    # Init State
    m_i = np.full((B, H, M, 1), -np.inf) # Max
    l_i = np.zeros((B, H, M, 1))         # Sum
    
    # Loop over K/V blocks (The "Streaming" dimension)
    for j in range(0, N, tile_size):
        # Load Tiles
        K_j = K[:, :, j:j+tile_size, :]
        V_j = V[:, :, j:j+tile_size, :]
        
        # 1. Compute Local QK^T (S_ij)
        S_ij = np.matmul(Q, K_j.transpose(0, 1, 3, 2)) * scale
        
        # 2. Update Online Softmax State
        m_ij = np.max(S_ij, axis=-1, keepdims=True)
        p_ij = np.exp(S_ij - m_ij)
        l_ij = np.sum(p_ij, axis=-1, keepdims=True)
        
        # 3. Rescaling Factor for previous Accumulator
        m_new = np.maximum(m_i, m_ij)
        alpha = np.exp(m_i - m_new)
        beta  = np.exp(m_ij - m_new)
        
        # 4. Update State and Accumulator
        # l_new = l_prev * alpha + l_current * beta
        l_new = l_i * alpha + l_ij * beta
        
        # O_new = O_prev * alpha + P_current * V_current * beta
        # Note: In standard online softmax derivation, P_current is already scaled by beta implicitly if we use m_new
        # Let's stick to the FlashAttention V1 derivation:
        # P_tilde_ij = exp(S_ij - m_new) = exp(S_ij - m_ij) * exp(m_ij - m_new) = p_ij * beta
        
        P_tilde_ij = p_ij * beta
        w_v = np.matmul(P_tilde_ij, V_j)
        
        O = O * alpha + w_v
        
        # Update running stats
        m_i = m_new
        l_i = l_new

    # Final Division
    O = O / l_i
    return O

def main():
    print("=== FMHA Step 1: Python Prototype ===")
    
    # Config
    B, H, M, N, D = 1, 4, 1024, 1024, 64
    np.random.seed(42)
    
    Q = np.random.randn(B, H, M, D).astype(np.float32)
    K = np.random.randn(B, H, N, D).astype(np.float32)
    V = np.random.randn(B, H, N, D).astype(np.float32)
    
    print(f"Shape: B={B}, H={H}, Seq={M}x{N}, D={D}")
    
    # 1. Run Standard
    print("Running Standard Attention...", end="")
    start = time.time()
    ref_O = standard_attention(Q, K, V)
    print(f" Done ({time.time()-start:.4f}s)")
    
    # 2. Run Online
    print("Running Online Softmax (Tiled)...", end="")
    start = time.time()
    res_O = online_softmax_attention(Q, K, V, tile_size=128)
    print(f" Done ({time.time()-start:.4f}s)")
    
    # 3. Verify
    print("Verifying Invariant...")
    max_diff = np.abs(ref_O - res_O).max()
    print(f"Max Error: {max_diff:.2e}")
    
    passed = bool(np.allclose(ref_O, res_O, atol=1e-5))
    if passed:
        print("✅ Invariant Check Passed: OnlineSoftmax == NativeSoftmax")
    else:
        print("❌ Invariant Check Failed")

    # DSL Trace Emission
    import json
    trace = {
        "type": "Correctness",
        "step_name": "Step 1: Python Prototype",
        "passed": passed,
        "max_error": float(max_diff),
        "component": "Softmax"
    }
    print(f"__SPAK_TRACE__{json.dumps(trace)}")

import time
if __name__ == "__main__":
    main()