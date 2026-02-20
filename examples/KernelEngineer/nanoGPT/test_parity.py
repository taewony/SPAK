import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from model import GPT as GPT_Original, GPTConfig as Config_Original
from nanogpt_cutile import GPT as GPT_CuTile, GPTConfig as Config_CuTile, _run_layernorm_static

def test_layernorm_parity():
    print("Testing LayerNorm Parity...")
    B, T, C = 2, 16, 384
    x = torch.randn(B, T, C, device='cuda', dtype=torch.float16)
    weight = torch.ones(C, device='cuda', dtype=torch.float16)
    bias = torch.zeros(C, device='cuda', dtype=torch.float16)
    y_ref = F.layer_norm(x, (C,), weight, bias, 1e-5)
    y_ct = _run_layernorm_static(x, weight, bias)
    diff = (y_ref - y_ct).abs().max().item()
    print(f"  Max Diff: {diff:.6e}")
    if diff < 1e-2:
        print("  LayerNorm Parity [PASS]")
    else:
        print("  LayerNorm Parity [FAIL]")

def test_attention_parity():
    print("Testing Attention Parity...")
    B, H, T, D = 2, 6, 64, 64 
    q = torch.randn(B, H, T, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H, T, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H, T, D, device='cuda', dtype=torch.float16)
    y_ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    from nanogpt_cutile import nanogpt_attention_kernel
    import cuda.tile as ct
    out_padded = torch.empty_like(q)
    scale_log2 = (1.0 / math.sqrt(D)) * (1.0 / math.log(2))
    grid = (1, B * H, 1)
    ct.launch(torch.cuda.current_stream(), grid, nanogpt_attention_kernel,
             (q, k, v, out_padded, scale_log2, -1e20, D, H, 64, 64, 2, 5))
    y_ct = out_padded
    diff = (y_ref - y_ct).abs().max().item()
    print(f"  Max Diff: {diff:.6e}")
    if diff < 5e-2:
        print("  Attention Parity [PASS]")
    else:
        print("  Attention Parity [FAIL]")

def test_model_forward_parity():
    print("Testing Full Model Forward Parity...")
    config_args = dict(n_layer=2, n_head=6, n_embd=384, block_size=256, vocab_size=65, bias=False)
    conf_orig = Config_Original(**config_args)
    model_orig = GPT_Original(conf_orig).cuda().eval()
    conf_ct = Config_CuTile(**config_args)
    model_ct = GPT_CuTile(conf_ct).cuda().eval()
    model_ct.load_state_dict(model_orig.state_dict())
    x = torch.randint(0, 65, (1, 10), device='cuda')
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            logits_ref, _ = model_orig(x)
            logits_ct, _ = model_ct(x)
    diff = (logits_ref - logits_ct).abs().max().item()
    print(f"  Max Logits Diff: {diff:.6e}")
    if diff < 1e-1:
        print("  Model Forward Parity [PASS]")
    else:
        print("  Model Forward Parity [FAIL]")

if __name__ == "__main__":
    test_layernorm_parity()
    test_attention_parity()
    test_model_forward_parity()
    print("ALL TESTS COMPLETED")
