import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

def apply_rope(q, k, cos, sin):
    # q, k: (B, nh, T, hs), cos, sin: (1, 1, T, hs)
    q_half1 = q[..., :q.shape[-1]//2]
    q_half2 = q[..., q.shape[-1]//2:]
    q_rotated = torch.cat((-q_half2, q_half1), dim=-1)
    q = (q * cos) + (q_rotated * sin)
    
    k_half1 = k[..., :k.shape[-1]//2]
    k_half2 = k[..., k.shape[-1]//2:]
    k_rotated = torch.cat((-k_half2, k_half1), dim=-1)
    k = (k * cos) + (k_rotated * sin)
    return q, k

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        
        # RoPE Frequencies
        head_dim = config.n_embd // config.n_head
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, cos=None, sin=None):
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        hs = C // self.n_head
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)

        if cos is not None and sin is not None:
            q, k = apply_rope(q, k, cos, sin)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # Fallback manual attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # Note: We skip the explicit bias buffer here for simplicity, assuming Flash if possible
            att = F.softmax(att, dim=-1)
            y = att @ v
            
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, cos=None, sin=None):
        x = x + self.attn(self.ln_1(x), cos=cos, sin=sin)
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight 
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def _get_cos_sin(self, T, device, dtype):
        head_dim = self.config.n_embd // self.config.n_head
        inv_freq = self.transformer.h[0].attn.inv_freq
        t = torch.arange(T, device=device, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)
        cos = freqs.cos().view(1, 1, T, head_dim).to(dtype)
        sin = freqs.sin().view(1, 1, T, head_dim).to(dtype)
        return cos, sin

    def forward(self, idx, targets=None, thinking_token_id=None):
        device = idx.device
        b, t = idx.size()
        tok_emb = self.transformer.wte(idx)
        x = self.transformer.drop(tok_emb)
        cos, sin = self._get_cos_sin(t, device, tok_emb.dtype)
        for block in self.transformer.h:
            x = block(x, cos=cos, sin=sin)
        x = self.transformer.ln_f(x)
        
        if targets is not None:
            logits = self.lm_head(x)
            
            # Efficient Loss Masking
            mask = torch.ones_like(targets, dtype=torch.float32)
            if thinking_token_id is not None:
                for i in range(b):
                    eq_indices = (idx[i] == thinking_token_id).nonzero(as_tuple=True)[0]
                    if len(eq_indices) > 0:
                        first_eq = eq_indices[0].item()
                        mask[i, :first_eq] = 0.0
            
            mask = mask.view(-1)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none', ignore_index=-1)
            loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss
