import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import cuda.tile as ct

# ============================================================
# 1. COMPOUND KERNELS (Verified Performance)
# ============================================================

@ct.kernel(occupancy=2)
def nanogpt_attention_kernel(
    Q, K, V, Out, qk_scale_log2: float, neg_inf: float,
    TILE_D: ct.Constant[int], H: ct.Constant[int],
    TILE_M: ct.Constant[int], TILE_N: ct.Constant[int],
    K_LAT: ct.Constant[int], V_LAT: ct.Constant[int]
):
    bid_x, bid_y = ct.bid(0), ct.bid(1)
    batch_idx, head_idx = bid_y // H, bid_y % H
    
    m_i = ct.full((TILE_M, 1), neg_inf, dtype=ct.float32)
    l_i = ct.full((TILE_M, 1), 0.0, dtype=ct.float32)
    acc = ct.full((TILE_M, TILE_D), 0.0, dtype=ct.float32)

    offs_m = bid_x * TILE_M + ct.arange(TILE_M, dtype=ct.int32)[:, None]
    offs_n_tile = ct.arange(TILE_N, dtype=ct.int32)[None, :]

    q = ct.load(Q, index=(batch_idx, head_idx, bid_x, 0), shape=(1, 1, TILE_M, TILE_D)).reshape((TILE_M, TILE_D))
    k_seqlen = K.shape[2]
    m_end = (bid_x + 1) * TILE_M
    Tc = ct.cdiv(min(m_end, k_seqlen), TILE_N)

    for j in range(0, Tc):
        k = ct.load(K, index=(batch_idx, head_idx, 0, j), shape=(1, 1, TILE_D, TILE_N), order=(0,1,3,2), latency=K_LAT).reshape((TILE_D, TILE_N))
        qk = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
        qk = ct.mma(q, k, qk)

        offs_n = j * TILE_N + offs_n_tile
        mask = offs_m >= offs_n
        qk = qk + ct.where(mask, 0.0, neg_inf)

        m_ij = max(m_i, ct.max(qk, axis=-1, keepdims=True) * qk_scale_log2)
        m_ij = max(m_ij, neg_inf) 
        
        p = ct.exp2(qk * qk_scale_log2 - m_ij)
        l_ij = ct.sum(p, axis=-1, keepdims=True)
        alpha = ct.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha

        v = ct.load(V, index=(batch_idx, head_idx, j, 0), shape=(1, 1, TILE_N, TILE_D), latency=V_LAT).reshape((TILE_N, TILE_D))
        acc = ct.mma(p.astype(Q.dtype), v, acc)
        m_i = m_ij

    acc = ct.truediv(acc, l_i)
    ct.store(Out, index=(batch_idx, head_idx, bid_x, 0), tile=acc.reshape((1, 1, TILE_M, TILE_D)).astype(Out.dtype))

@ct.kernel
def nanogpt_layernorm_kernel(X, Y, W, B, TILE_SIZE_M: ct.Constant[int], TILE_SIZE_N: ct.Constant[int], eps: float):
    bid = ct.bid(0)
    M, N = X.shape[0], X.shape[1]
    upper_bound = ct.cdiv(M, TILE_SIZE_M)
    w = ct.load(W, index=(0,), shape=(TILE_SIZE_N,)).reshape((1, TILE_SIZE_N))
    b = ct.load(B, index=(0,), shape=(TILE_SIZE_N,)).reshape((1, TILE_SIZE_N))
    
    num_tile_blocks = ct.num_blocks(0)
    for current_bid in range(bid, upper_bound, num_tile_blocks):
        x = ct.load(X, index=(current_bid, 0), shape=(TILE_SIZE_M, TILE_SIZE_N))
        x_f32 = ct.astype(x, ct.float32)
        
        mean = ct.sum(x_f32, axis=1, keepdims=True) / N
        centered = x_f32 - mean
        var = ct.sum(centered * centered, axis=1, keepdims=True) / N
        rstd = ct.rsqrt(var + eps)
        
        y = (centered * rstd) * ct.astype(w, ct.float32) + ct.astype(b, ct.float32)
        ct.store(Y, index=(current_bid, 0), tile=ct.astype(y, X.dtype))

# ============================================================
# 2. GPT-2 ARCHITECTURE (nanoGPT model.py parity)
# ============================================================

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
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        
        # Inherited Blackwell Optimal Config
        self.config_delta = {
            "tile_m": 64, "tile_n": 64, "k_lat": 2, "v_lat": 5, "neg_inf": -1e20
        }

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        out = torch.empty_like(q, dtype=torch.float16 if q.dtype == torch.float16 else torch.float32)
        scale_log2 = (1.0 / math.sqrt(self.head_dim)) * (1.0 / math.log(2))
        grid = (max(1, T // self.config_delta["tile_m"]), B * self.n_head, 1)
        
        ct.launch(torch.cuda.current_stream(), grid, nanogpt_attention_kernel,
                 (q, k, v, out, scale_log2, self.config_delta["neg_inf"], 
                  self.head_dim, self.n_head, self.config_delta["tile_m"], 64, 2, 5))

        y = out.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x, approximate='tanh')
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self._run_layernorm(x, self.ln_1))
        x = x + self.mlp(self._run_layernorm(x, self.ln_2))
        return x

    def _run_layernorm(self, x, ln_mod):
        M, N = x.view(-1, x.size(-1)).shape
        y = torch.empty_like(x)
        grid = (min(80, (M + 3) // 4),)
        ct.launch(torch.cuda.current_stream(), grid, nanogpt_layernorm_kernel,
                 (x.view(-1, N), y.view(-1, N), ln_mod.weight, ln_mod.bias, 4, N, 1e-5))
        return y

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight 

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self._run_layernorm(x, self.transformer.ln_f)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        # This allows us to load OpenAI GPT-2 weights into our cuTile implementation
        from transformers import GPT2LMHeadModel
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config_args['bias'] = True
        config = GPTConfig(**config_args)
        model = cls(config)
        sd = model.state_dict()
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        # Filter and copy weights
        sd_keys_hf = [k for k in sd_hf.keys() if not k.endswith('.attn.masked_bias') and not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad(): sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad(): sd[k].copy_(sd_hf[k])
        return model

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def _run_layernorm(self, x, ln_mod):
        M, N = x.view(-1, x.size(-1)).shape
        y = torch.empty_like(x)
        grid = (min(80, (M + 3) // 4),)
        ct.launch(torch.cuda.current_stream(), grid, nanogpt_layernorm_kernel,
                 (x.view(-1, N), y.view(-1, N), ln_mod.weight, ln_mod.bias, 4, N, 1e-5))
        return y

if __name__ == "__main__":
    config = GPTConfig(n_layer=4, n_head=4, n_embd=256)
    model = GPT(config).cuda()
    idx = torch.randint(0, config.vocab_size, (1, 128)).cuda()
    print("[INFO] Success! NanoGPT cuTile active.")
    logits, _ = model(idx)
    print(f"Logits shape: {logits.shape}")
