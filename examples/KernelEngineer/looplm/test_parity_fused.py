import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model import Block, GPTConfig
from model_loop import LoopGPT

def test_fused_parity():
    print("Testing LoopLM Fused Kernel Parity...")
    device = 'cuda'
    n_embd = 256
    n_head = 4
    vocab_size = 13
    block_size = 256
    
    config = GPTConfig(n_embd=n_embd, n_head=n_head, vocab_size=vocab_size, block_size=block_size, bias=False)
    # We test with inject_x0=True as it's our mandatory architecture
    model = LoopGPT(config, num_loops=1, inject_x0=True).to(device).eval()
    
    # Input: 1 batch, 10 tokens
    x = torch.randint(0, vocab_size, (1, 10), device=device)
    
    # 1. Get Reference Output (Standard Path)
    # To test the internal kernel, we must bypass the kernel in model_loop by setting halt_threshold=None
    # But wait, the kernel IS the inference path now if halt_threshold is set.
    
    with torch.no_grad():
        # Path A: PyTorch Reference (Manual block call)
        pos = torch.arange(0, 10, dtype=torch.long, device=device)
        tok_emb = model.transformer.wte(x)
        pos_emb = model.transformer.wpe(pos)
        x0 = model.transformer.drop(tok_emb + pos_emb)
        
        # Standard LoopGPT training-style step (Equivalent to what we want)
        h_input = x0 + x0 # because h_curr starts at x0 and inject_x0 is True
        h_ref = model.transformer.h(h_input + model.step_embedding(torch.tensor([0], device=device)))
        logits_ref = model.lm_head(model.transformer.ln_f(h_ref))

        # Path B: Fused Kernel Path
        # We trigger the kernel by providing a halt_threshold
        logits_kern, _, steps = model(x, num_loops=1, halt_threshold=0.9)
        
    # 2. Compare
    # Note: Currently our fused kernel only does h + res, so we expect a difference 
    # until we implement the full MLP inside the kernel.
    diff_logits = (logits_ref[:, -1, :] - logits_kern.view(-1)).abs().max().item()
    print(f"  Max Logit Difference: {diff_logits:.6e}")
    
    if diff_logits < 1e-3:
        print("  ✅ [PASS] Kernel Parity confirmed.")
    else:
        print("  ❌ [FAIL] Kernel Parity mismatch. Logic update required.")

if __name__ == "__main__":
    test_fused_parity()
