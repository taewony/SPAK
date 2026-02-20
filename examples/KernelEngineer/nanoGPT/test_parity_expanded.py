import torch
import torch.nn.functional as F
import math
import os
from model import GPT as GPT_Original, GPTConfig as Config_Original
from nanogpt_cutile import GPT as GPT_CuTile, GPTConfig as Config_CuTile, _run_layernorm_static

def check(name, diff, threshold=1e-2):
    status = "[PASS]" if diff < threshold else "[FAIL]"
    print(f"{name}: Max Diff = {diff:.6e} {status}")
    if diff >= threshold:
        print(f"  WARNING: {name} exceeds threshold {threshold}")

def test_layernorm_suite():
    print("--- Testing LayerNorm Suite ---")
    device = 'cuda'
    # Base Case: Standard 384 dim
    B, T, C = 2, 16, 384
    x = torch.randn(B, T, C, device=device, dtype=torch.float16)
    w = torch.ones(C, device=device, dtype=torch.float16)
    b = torch.zeros(C, device=device, dtype=torch.float16)
    y_ref = F.layer_norm(x, (C,), w, b, 1e-5)
    y_ct = _run_layernorm_static(x, w, b)
    check("LN Base (T=16, C=384)", (y_ref - y_ct).abs().max().item())

    # Expanded Case 1: Non-multiple of 4 rows
    B, T, C = 1, 13, 384
    x = torch.randn(B, T, C, device=device, dtype=torch.float16)
    w = torch.randn(C, device=device, dtype=torch.float16)
    b = torch.randn(C, device=device, dtype=torch.float16)
    y_ref = F.layer_norm(x, (C,), w, b, 1e-5)
    y_ct = _run_layernorm_static(x, w, b)
    check("LN Case 1 (T=13, Random W/B)", (y_ref - y_ct).abs().max().item())

    # Expanded Case 2: Large M and 768 dim
    B, T, C = 32, 128, 768
    x = torch.randn(B, T, C, device=device, dtype=torch.float16)
    w = torch.ones(C, device=device, dtype=torch.float16)
    b = torch.zeros(C, device=device, dtype=torch.float16)
    y_ref = F.layer_norm(x, (C,), w, b, 1e-5)
    y_ct = _run_layernorm_static(x, w, b)
    check("LN Case 2 (Large batch, C=768)", (y_ref - y_ct).abs().max().item())

def test_attention_suite():
    print("--- Testing Attention Suite ---")
    device = 'cuda'
    from nanogpt_cutile import nanogpt_attention_kernel
    import cuda.tile as ct
    def run_attn_test(B, H, T, D, name):
        q = torch.randn(B, H, T, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H, T, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H, T, D, device=device, dtype=torch.float16)
        y_ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        tile_m, tile_n = 64, 64
        T_padded = math.ceil(T / tile_m) * tile_m
        out_padded = torch.empty((B, H, T_padded, D), dtype=q.dtype, device=device)
        scale_log2 = (1.0 / math.sqrt(D)) * (1.0 / math.log(2))
        # Ensure grid_x is at least 1
        grid_x = max(1, T_padded // tile_m)
        grid = (grid_x, B * H, 1)
        ct.launch(torch.cuda.current_stream(), grid, nanogpt_attention_kernel,
                 (q, k, v, out_padded, scale_log2, -1e20, D, H, tile_m, tile_n, 2, 5))
        y_ct = out_padded[:, :, :T, :]
        diff = (y_ref - y_ct).abs().max().item()
        check(name, diff, threshold=5e-2)
    
    run_attn_test(2, 6, 64, 64, "Attn Base (T=64)")
    run_attn_test(1, 6, 13, 64, "Attn Case 1 (T=13)")
    run_attn_test(1, 6, 127, 64, "Attn Case 2 (T=127)")
    run_attn_test(1, 6, 257, 64, "Attn Case 3 (T=257)")

def test_full_model_suite():
    print("--- Testing Full Model Forward Suite ---")
    device = 'cuda'
    # Level 1: Small debug model
    config_small = dict(n_layer=2, n_head=6, n_embd=384, block_size=256, vocab_size=65, bias=False)
    m_orig_s = GPT_Original(Config_Original(**config_small)).to(device).eval()
    m_ct_s = GPT_CuTile(Config_CuTile(**config_small)).to(device).eval()
    m_ct_s.load_state_dict(m_orig_s.state_dict())
    x_s = torch.randint(0, 65, (1, 10), device=device)
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            l_ref_s, _ = m_orig_s(x_s)
            l_ct_s, _ = m_ct_s(x_s)
    check("Small Model Forward (L=2, D=384)", (l_ref_s[:, -1, :] - l_ct_s[:, -1, :]).abs().max().item(), threshold=1e-1)

    # Level 2: GPT-2 Small Config
    config_full = dict(n_layer=12, n_head=12, n_embd=768, block_size=1024, vocab_size=65, bias=False)
    m_orig_f = GPT_Original(Config_Original(**config_full)).to(device).eval()
    m_ct_f = GPT_CuTile(Config_CuTile(**config_full)).to(device).eval()
    m_ct_f.load_state_dict(m_orig_f.state_dict())
    x_f = torch.randint(0, 65, (1, 150), device=device)
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            l_ref_f, _ = m_orig_f(x_f)
            l_ct_f, _ = m_ct_f(x_f)
    check("Full Model Forward (L=12, D=768)", (l_ref_f[:, -1, :] - l_ct_f[:, -1, :]).abs().max().item(), threshold=1e-1)

if __name__ == "__main__":
    test_layernorm_suite()
    test_attention_suite()
    test_full_model_suite()
    print("ALL CONSOLIDATED TESTS COMPLETED")
