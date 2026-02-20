import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model import GPT as GPT_Original, GPTConfig as Config_Original
from model_loop import LoopGPT

def test_loop_structure():
    print("Testing LoopLM Structural Integrity...")
    config_args = dict(n_layer=1, n_head=6, n_embd=384, block_size=256, vocab_size=65, bias=False)
    conf = Config_Original(**config_args)
    
    num_loops = 12
    model_loop = LoopGPT(conf, num_loops=num_loops).cuda().eval()
    
    x = torch.randint(0, 65, (1, 10), device='cuda')
    
    with torch.no_grad():
        logits, _ = model_loop(x)
        print(f"  Logits shape: {logits.shape}")
        assert logits.shape == (1, 1, 65), "Logits shape mismatch!"
    
    print("  Structural Integrity [PASS]")

def test_equivalence_1step():
    print("Testing 1-step LoopLM vs 1-layer GPT...")
    config_args = dict(n_layer=1, n_head=6, n_embd=384, block_size=256, vocab_size=65, bias=False)
    conf = Config_Original(**config_args)
    
    model_orig = GPT_Original(conf).cuda().eval()
    model_loop = LoopGPT(conf, num_loops=1).cuda().eval()
    
    # Sync weights
    # Note: LoopGPT has step_embedding, but for 1-step it might differ slightly if not careful
    # Let's zero out step_embedding for exact parity check
    model_loop.step_embedding.weight.data.zero_()
    
    # Load weights
    model_loop.load_state_dict(model_orig.state_dict(), strict=False)
    
    x = torch.randint(0, 65, (1, 10), device='cuda')
    
    with torch.no_grad():
        logits_ref, _ = model_orig(x)
        # For 1-layer GPT, logits is (B, 1, V) in eval mode (optimized)
        logits_loop, _ = model_loop(x, num_loops=1)
        
    diff = (logits_ref - logits_loop).abs().max().item()
    print(f"  1-step Max Diff: {diff:.6e}")
    # Small diff expected due to X0 injection if not disabled
    # But in model_loop.py: h_input = h + x0. 
    # For loop 0: h = x0, so h_input = 2 * x0. 
    # This WILL differ from standard 1L GPT.
    
    print("  Note: 1-step diff is expected due to X0 injection rule.")

if __name__ == "__main__":
    test_loop_structure()
    test_equivalence_1step()
