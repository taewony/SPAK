import torch
import torch.nn.functional as F
import math
import os
from model import GPT as GPT_Original, GPTConfig as Config_Original
from nanogpt_cutile import GPT as GPT_CuTile, _run_layernorm_static

def check(name, diff, threshold=1e-2):
    status = "[PASS]" if diff < threshold else "[FAIL]"
    print(f"{name}: Max Diff = {diff:.6e} {status}")
    if diff >= threshold:
        print(f"  WARNING: {name} exceeds threshold {threshold}")

def test_layernorm_expanded():
    print("
--- Testing LayerNorm Expanded Scope ---")
    device = 'cuda'
    # Case 1: Non-power-of-two, non-multiple of 4 rows
    B, T, C = 1, 13, 384
    x = torch.randn(B, T, C, device=device, dtype=torch.float16)
    w = torch.randn(C, device=device, dtype=torch.float16)
    b = torch.randn(C, device=device, dtype=torch.float16)
    
    y_ref = F.layer_norm(x, (C,), w, b, 1e-5)
    y_ct = _run_layernorm_static(x, w, b)
    check("LN Case 1 (T=13, Random W/B)", (y_ref - y_ct).abs().max().item())

    # Case 2: Large M (many rows)
    B, T, C = 32, 128, 768
    x = torch.randn(B, T, C, device=device, dtype=torch.float16)
    w = torch.ones(C, device=device, dtype=torch.float16)
    b = torch.zeros(C, device=device, dtype=torch.float16)
    y_ref = F.layer_norm(x, (C,), w, b, 1e-5)
    y_ct = _run_layernorm_static(x, w, b)
    check("LN Case 2 (Large batch, C=768)", (y_ref - y_ct).abs().max().item())

def test_attention_expanded():
    print("
--- Testing Attention Expanded Scope ---")
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
        grid = (T_padded // tile_m, B * H, 1)
        
        ct.launch(torch.cuda.current_stream(), grid, nanogpt_attention_kernel,
                 (q, k, v, out_padded, scale_log2, -1e20, D, H, tile_m, tile_n, 2, 5))
        
        y_ct = out_padded[:, :, :T, :]
        diff = (y_ref - y_ct).abs().max().item()
        check(name, diff, threshold=5e-2)

    # Case 1: T < Tile Size (Padding check)
    run_attn_test(1, 6, 13, 64, "Attn Case 1 (T=13)")
    # Case 2: T > Tile Size, not multiple (Multi-tile + Padding check)
    run_attn_test(1, 6, 127, 64, "Attn Case 2 (T=127)")
    # Case 3: T very large (Boundary check)
    run_attn_test(1, 6, 257, 64, "Attn Case 3 (T=257)")

def test_full_model_parity():
    print("
--- Testing Full Model Forward (GPT-2 Small Config) ---")
    device = 'cuda'
    # GPT-2 Small: 12 layers, 768 dim, 12 heads
    config_args = dict(n_layer=12, n_head=12, n_embd=768, block_size=1024, vocab_size=65, bias=False)
    
    conf_orig = Config_Original(**config_args)
    model_orig = GPT_Original(conf_orig).to(device).eval()
    
    conf_ct = Config_CuTile(**config_args)
    model_ct = GPT_CuTile(conf_ct).to(device).eval()
    
    # Sync weights exactly
    model_ct.load_state_dict(model_orig.state_dict())
    
    # Test with sequence that crosses multiple tiles
    T_test = 150 
    x = torch.randint(0, 65, (1, T_test), device=device)
    
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            logits_ref, _ = model_orig(x)
            logits_ct, _ = model_ct(x)
            
    # Logits are (B, T, V) or (B, 1, V) depending on forward() optimization
    # Our optimized forward returns (B, 1, V) for inference.
    # Let's adjust reference to compare the last token.
    diff = (logits_ref[:, -1, :] - logits_ct[:, -1, :]).abs().max().item()
    check("Full Model Logits (T=150, L=12, D=768)", diff, threshold=1e-1)

if __name__ == "__main__":
    test_layernorm_expanded()
    test_attention_expanded()
    test_full_model_parity()
    print("
EXPANDED PARITY TESTS COMPLETED")
