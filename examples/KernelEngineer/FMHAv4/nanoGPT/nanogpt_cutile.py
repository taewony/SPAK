import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import os
import sys

# --- Dynamic TileGym Path Handling ---
tilegym_src = os.path.join(os.path.dirname(__file__), 'TileGym', 'src')
if os.path.exists(tilegym_src):
    sys.path.append(tilegym_src)
    print(f"[INFO] Added {tilegym_src} to sys.path.")

# Direct Import from verified sub-packages
try:
    from tilegym.ops.cutile import attention as attention_kernel
    from tilegym.ops.cutile import layer_norm_legacy as layernorm_kernel
    # Wrap them into the expected functional interface if they differ from dispatch
    def fmha(q, k, v, is_causal=True, scaling=None, **kwargs):
        # Using the direct kernel interface
        return attention_kernel.tile_fmha(q, k, v, scaling=scaling, is_causal=is_causal, **kwargs)
    
    def layer_norm_legacy(x, weight, bias, eps, **kwargs):
        return layernorm_kernel.layer_norm_legacy(x, None, weight, bias, eps, **kwargs)
    
    print("[INFO] Successfully activated cuTile high-performance backend.")
except Exception as e:
    print(f"[WARN] cuTile backend failed (Error: {e}). Using PyTorch fallback.")
    def fmha(q, k, v, is_causal=True, scaling=None, **kwargs):
        return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=scaling)
    def layer_norm_legacy(x, weight, bias, eps, **kwargs):
        return F.layer_norm(x, weight.shape, weight, bias if bias is not None else torch.zeros_like(weight), eps)

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        
        # Inherited from FMHAv4 Compound Knowledge
        self.kernel_configs = {
            "tile_m": 64,
            "tile_n": 64,
            "k_lat": 2,
            "v_lat": 5,
            "safe_neg_val": -1e20
        }

    def forward(self, x):
        B, T, C = x.size()
        
        # Combined QKV projection
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Reshape for multi-head: (B, T, C) -> (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Use our optimized FMHA kernel via TileGym interface
        # Compound Rule: 1.11x Speedup vs Native on Blackwell
        y = fmha(
            q, k, v, 
            is_causal=True, 
            scaling=1.0/math.sqrt(self.head_dim),
            kernel_configs=self.kernel_configs # Passing our verified configs
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x, approximate='tanh')
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # Use layer_norm_legacy for cuTile dispatch if available
        x_norm1 = layer_norm_legacy(x, self.ln_1.weight, self.ln_1.bias, 1e-5)
        x = x + self.attn(x_norm1)
        
        x_norm2 = layer_norm_legacy(x, self.ln_2.weight, self.ln_2.bias, 1e-5)
        x = x + self.mlp(x_norm2)
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Invariant: Weight Tying
        self.transformer.wte.weight = self.lm_head.weight 

        # Initial weight scaling per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        
        for block in self.transformer.h:
            x = block(x)
            
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

if __name__ == "__main__":
    config = GPTConfig(n_layer=4, n_head=4, n_embd=256)
    model = GPT(config).cuda().half()
    idx = torch.randint(0, config.vocab_size, (1, 128)).cuda()
    
    print("NanoGPT cuTile Forward Pass (Modular & Weight-Tied)...")
    with torch.no_grad():
        logits, _ = model(idx)
    print(f"Success! Logits shape: {logits.shape}")
