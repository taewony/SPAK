import torch
import torch.nn as nn
import math
from microgpt_cutile import Block, MicroGPT

class ReferenceBlock(nn.Module):
    """ Pure PyTorch implementation of the same logic for sanity checking """
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.ln1_weight = nn.Parameter(torch.ones(n_embd))
        self.ln2_weight = nn.Parameter(torch.ones(n_embd))
        self.wq = nn.Linear(n_embd, n_embd, bias=False)
        self.wk = nn.Linear(n_embd, n_embd, bias=False)
        self.wv = nn.Linear(n_embd, n_embd, bias=False)
        self.wo = nn.Linear(n_embd, n_embd, bias=False)
        self.fc1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.fc2 = nn.Linear(4 * n_embd, n_embd, bias=False)

    def _rmsnorm(self, x, weight):
        ms = x.pow(2).mean(-1, keepdim=True)
        return (x * torch.rsqrt(ms + 1e-5)) * weight

    def forward(self, x):
        B, T, C = x.shape
        # Attention
        x_norm = self._rmsnorm(x, self.ln1_weight)
        q = self.wq(x_norm).view(B, T, self.n_head, -1).transpose(1, 2)
        k = self.wk(x_norm).view(B, T, self.n_head, -1).transpose(1, 2)
        v = self.wv(x_norm).view(B, T, self.n_head, -1).transpose(1, 2)
        
        # Reference Causal SDPA
        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).reshape(B, T, -1)
        x = x + self.wo(attn_out)
        
        # MLP
        x_norm = self._rmsnorm(x, self.ln2_weight)
        mlp_h = torch.relu(self.fc1(x_norm))
        x = x + self.fc2(mlp_h)
        return x

def check_parity():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("[WARN] No CUDA device found. Can only check model initialization, not kernel execution.")
    
    n_embd, n_head = 64, 4
    cutile_block = Block(n_embd, n_head).to(device).half()
    ref_block = ReferenceBlock(n_embd, n_head).to(device).half()
    
    # Sync weights exactly
    with torch.no_grad():
        ref_block.ln1_weight.copy_(cutile_block.ln1)
        ref_block.ln2_weight.copy_(cutile_block.ln2)
        ref_block.wq.weight.copy_(cutile_block.wq.weight)
        ref_block.wk.weight.copy_(cutile_block.wk.weight)
        ref_block.wv.weight.copy_(cutile_block.wv.weight)
        ref_block.wo.weight.copy_(cutile_block.wo.weight)
        ref_block.fc1.weight.copy_(cutile_block.fc1.weight)
        ref_block.fc2.weight.copy_(cutile_block.fc2.weight)

    x = torch.randn(1, 16, n_embd, device=device).half()
    
    print("--- Sanity Check: cuTile vs. Reference ---")
    try:
        with torch.no_grad():
            expected = ref_block(x)
            if device == 'cuda':
                actual = cutile_block(x)
                diff = (actual - expected).abs().max().item()
                print(f"Max Difference: {diff:.6f}")
                if diff < 1e-2:
                    print("[PASS] Block parity verified.")
                else:
                    print("[FAIL] Significant divergence detected!")
            else:
                print("[SKIP] Kernel execution skipped (no CUDA).")
    except Exception as e:
        print(f"[ERROR] Logic check failed: {e}")

if __name__ == "__main__":
    check_parity()
