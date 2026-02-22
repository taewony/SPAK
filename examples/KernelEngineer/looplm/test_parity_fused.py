import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model import Block, GPTConfig
from model_loop import LoopGPT

def test_fused_parity():
    print("Testing LoopLM Fused Kernel Parity (Step-by-Step)...")
    device = 'cuda'
    n_embd = 256
    n_head = 4
    vocab_size = 13
    block_size = 256
    
    config = GPTConfig(n_embd=n_embd, n_head=n_head, vocab_size=vocab_size, block_size=block_size)
    model = LoopGPT(config, num_loops=1).to(device).eval()
    
    x = torch.randint(0, vocab_size, (1, 10), device=device)
    
    with torch.no_grad():
        # Standard PyTorch Output (Baseline)
        logits_ref, _, steps_ref = model(x, num_loops=1, halt_threshold=None)
        
        print(f"  Reference Output Shape: {logits_ref.shape}")
        
        # In Phase 4.1, we'll replace the loop in model_loop.py with 
        # a single fused kernel call and verify here.
        
    print("  [Step 1] Initializing Parity Check... OK")
    print("  [Step 2] Next: Replacing attention/MLP loop with Fused Kernel.")

if __name__ == "__main__":
    test_fused_parity()
