import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import cuda.tile as ct

# --- Helper for power-of-two tiling ---
def next_pow2(n):
    return 2**(n - 1).bit_length()

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return _run_layernorm_static(x, self.weight, self.bias)

def _run_layernorm_static(x, weight, bias):
    orig_shape = x.shape
    x_flat = x.view(-1, x.size(-1))
    M, N = x_flat.shape
    
    tile_m = 4
    tile_n = next_pow2(N)
    M_padded = math.ceil(M / tile_m) * tile_m
    
    # Pad input, weight, bias to tile_n and M_padded
    x_padded = F.pad(x_flat, (0, tile_n - N, 0, M_padded - M))
    w_padded = F.pad(weight, (0, tile_n - N))
    # Handle optional bias
    if bias is not None:
        b_padded = F.pad(bias, (0, tile_n - N))
    else:
        b_padded = torch.zeros(tile_n, dtype=x.dtype, device=x.device)
    
    y_padded = torch.empty((M_padded, tile_n), dtype=x.dtype, device=x.device)
    
    grid = (min(80, M_padded // tile_m),)
    ct.launch(torch.cuda.current_stream(), grid, nanogpt_layernorm_kernel,
             (x_padded, y_padded, w_padded, b_padded, N, tile_m, tile_n, 1e-5))
    return y_padded[:M, :N].contiguous().view(orig_shape)

# ============================================================
# 1. OPTIMIZED KERNELS
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

    # Restore to tile-based indexing (bid_x)
    q = ct.load(Q, index=(batch_idx, head_idx, bid_x, 0), shape=(1, 1, TILE_M, TILE_D), padding_mode=ct.PaddingMode.ZERO).reshape((TILE_M, TILE_D))
    k_seqlen = K.shape[2]
    m_end = (bid_x + 1) * TILE_M
    Tc = ct.cdiv(min(m_end, k_seqlen), TILE_N)

    for j in range(0, Tc):
        # Restore to tile-based indexing (j)
        k = ct.load(K, index=(batch_idx, head_idx, 0, j), shape=(1, 1, TILE_D, TILE_N), order=(0,1,3,2), latency=K_LAT, padding_mode=ct.PaddingMode.ZERO).reshape((TILE_D, TILE_N))
        qk = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
        qk = ct.mma(q, k, qk)

        offs_n = j * TILE_N + offs_n_tile
        # Mask both causal and sequence length bounds
        mask = (offs_m >= offs_n) & (offs_n < k_seqlen)
        qk = qk + ct.where(mask, 0.0, neg_inf)

        # Numerical stability: multiply scale after max to match FMHAv4 and avoid overflow
        m_ij = max(m_i, ct.max(qk, axis=-1, keepdims=True) * qk_scale_log2)
        
        p = ct.exp2(qk * qk_scale_log2 - m_ij, flush_to_zero=True)
        l_ij = ct.sum(p, axis=-1, keepdims=True)
        alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)
        
        l_i = l_i * alpha + l_ij
        acc = acc * alpha

        # Restore to tile-based indexing (j)
        v = ct.load(V, index=(batch_idx, head_idx, j, 0), shape=(1, 1, TILE_N, TILE_D), latency=V_LAT, padding_mode=ct.PaddingMode.ZERO).reshape((TILE_N, TILE_D))
        acc = ct.mma(p.astype(Q.dtype), v, acc)
        m_i = m_ij

    acc = ct.truediv(acc, l_i)
    # Restore to tile-based indexing (bid_x)
    ct.store(Out, index=(batch_idx, head_idx, bid_x, 0), tile=acc.reshape((1, 1, TILE_M, TILE_D)).astype(Out.dtype))

@ct.kernel
def nanogpt_layernorm_kernel(
    X, Y, W, B, 
    N: ct.Constant[int], # Actual non-pow2 dimension
    TILE_SIZE_M: ct.Constant[int], 
    TILE_SIZE_N: ct.Constant[int], # Pow2 tile size >= N
    eps: float
):
    bid = ct.bid(0)
    M = X.shape[0]
    upper_bound = ct.cdiv(M, TILE_SIZE_M)
    
    # Load weights with padding
    w = ct.load(W, index=(0,), shape=(TILE_SIZE_N,), padding_mode=ct.PaddingMode.ZERO).reshape((1, TILE_SIZE_N))
    b = ct.load(B, index=(0,), shape=(TILE_SIZE_N,), padding_mode=ct.PaddingMode.ZERO).reshape((1, TILE_SIZE_N))
    
    # Create mask for mean/var calculation
    offs_n = ct.arange(TILE_SIZE_N, dtype=ct.int32)[None, :]
    mask = offs_n < N
    
    num_tile_blocks = ct.num_blocks(0)
    for current_bid in range(bid, upper_bound, num_tile_blocks):
        # Restore to tile-based indexing (current_bid)
        x = ct.load(X, index=(current_bid, 0), shape=(TILE_SIZE_M, TILE_SIZE_N), padding_mode=ct.PaddingMode.ZERO)
        x_f32 = ct.astype(x, ct.float32)
        
        # Masked mean
        sum_x = ct.sum(x_f32, axis=1, keepdims=True)
        mean = sum_x / N
        
        # Masked variance
        centered = x_f32 - mean
        centered_sq = centered * centered
        # We must zero out the squares of padded elements to not corrupt the variance sum
        centered_sq = ct.where(mask, centered_sq, 0.0)
        var = ct.sum(centered_sq, axis=1, keepdims=True) / N
        rstd = ct.rsqrt(var + eps)
        
        y = (centered * rstd) * ct.astype(w, ct.float32) + ct.astype(b, ct.float32)
        # Restore to tile-based indexing (current_bid)
        ct.store(Y, index=(current_bid, 0), tile=ct.astype(y, X.dtype))

# ============================================================
# 2. GPT-2 ARCHITECTURE
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
        self.config_delta = {"tile_m": 64, "tile_n": 64, "k_lat": 2, "v_lat": 5, "neg_inf": -1e20}

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Attention forward with padding for T to match TILE_M
        tile_m = self.config_delta["tile_m"]
        T_padded = math.ceil(T / tile_m) * tile_m
        out_padded = torch.empty((B, self.n_head, T_padded, self.head_dim), dtype=q.dtype, device=q.device)
        
        scale_log2 = (1.0 / math.sqrt(self.head_dim)) * (1.0 / math.log(2))
        grid_x = max(1, T_padded // tile_m)
        grid = (grid_x, B * self.n_head, 1)
        
        ct.launch(torch.cuda.current_stream(), grid, nanogpt_attention_kernel,
                 (q, k, v, out_padded, scale_log2, self.config_delta["neg_inf"], 
                  self.head_dim, self.n_head, tile_m, self.config_delta["tile_n"], 2, 5))

        out = out_padded[:, :, :T, :].contiguous()
        y = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x) # Match model.py exactly
        return self.dropout(self.c_proj(x))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
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
        x = self.transformer.ln_f(x)
        
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            # logits is (B, 1, V) due to optimization in forward()
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

if __name__ == "__main__":
    config = GPTConfig(n_layer=4, n_head=4, n_embd=256)
    model = GPT(config).cuda()
    idx = torch.randint(0, config.vocab_size, (1, 128)).cuda()
    print("[INFO] Success! NanoGPT cuTile active.")
    logits, _ = model(idx)
    print(f"Logits shape: {logits.shape}")
