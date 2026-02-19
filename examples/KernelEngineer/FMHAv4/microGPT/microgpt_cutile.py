import torch
import torch.nn as nn
import cuda.tile as ct
import math
import time

# --- Optimized Kernels ---

@ct.kernel(occupancy=2)
def microgpt_attention_kernel(
    Q, K, V, Out, qk_scale_log2: float,
    TILE_D: ct.Constant[int], H: ct.Constant[int],
    TILE_M: ct.Constant[int], TILE_N: ct.Constant[int],
    K_LAT: ct.Constant[int], V_LAT: ct.Constant[int]
):
    bid_x, bid_y = ct.bid(0), ct.bid(1)
    batch_idx, head_idx = bid_y // H, bid_y % H
    
    # Use float('inf') to avoid TypeInfer errors
    NEG_INF = -float('inf')
    
    m_i = ct.full((TILE_M, 1), NEG_INF, dtype=ct.float32)
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
        qk = qk + ct.where(mask, 0.0, NEG_INF)

        m_ij = max(m_i, ct.max(qk, axis=-1, keepdims=True) * qk_scale_log2)
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
def microgpt_rmsnorm_kernel(X, Y, W, TILE_SIZE_M: ct.Constant[int], TILE_SIZE_N: ct.Constant[int], eps: float):
    bid = ct.bid(0)
    M, N = X.shape[0], X.shape[1]
    upper_bound = ct.cdiv(M, TILE_SIZE_M)
    w = ct.load(W, index=(0,), shape=(TILE_SIZE_N,)).reshape((1, TILE_SIZE_N))
    
    num_tile_blocks = ct.num_blocks(0)
    for current_bid in range(bid, upper_bound, num_tile_blocks):
        x = ct.load(X, index=(current_bid, 0), shape=(TILE_SIZE_M, TILE_SIZE_N))
        x_f32 = ct.astype(x, ct.float32)
        var = ct.sum(x_f32 * x_f32, axis=1, keepdims=True) / N
        rstd = ct.rsqrt(var + eps)
        y = (x_f32 * rstd) * ct.astype(w, ct.float32)
        ct.store(Y, index=(current_bid, 0), tile=ct.astype(y, X.dtype))

# --- Helper for cuTile Launches ---

def run_rmsnorm_op(x, weight):
    M, N = x.reshape(-1, x.shape[-1]).shape
    y = torch.empty_like(x)
    # Use small TILE_SIZE_M=4 for stability with small batches
    grid = (min(80, ct.cdiv(M, 4)),)
    ct.launch(torch.cuda.current_stream(), grid, microgpt_rmsnorm_kernel, 
             (x.view(-1, N), y.view(-1, N), weight, 4, N, 1e-5))
    return y

# --- GPT Modules ---

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.ln1 = nn.Parameter(torch.ones(n_embd))
        self.ln2 = nn.Parameter(torch.ones(n_embd))
        self.wq = nn.Linear(n_embd, n_embd, bias=False)
        self.wk = nn.Linear(n_embd, n_embd, bias=False)
        self.wv = nn.Linear(n_embd, n_embd, bias=False)
        self.wo = nn.Linear(n_embd, n_embd, bias=False)
        self.fc1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.fc2 = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        # Attention path
        x_norm = run_rmsnorm_op(x, self.ln1)
        q = self.wq(x_norm).view(B, T, self.n_head, -1).transpose(1, 2)
        k = self.wk(x_norm).view(B, T, self.n_head, -1).transpose(1, 2)
        v = self.wv(x_norm).view(B, T, self.n_head, -1).transpose(1, 2)
        
        out = torch.empty_like(q)
        head_dim = q.size(-1)
        scale_log2 = (1.0 / math.sqrt(head_dim)) * (1.0 / math.log(2))
        
        # Use TILE_M=16 to match block_size, ensuring grid is at least 1
        tile_m = 16
        grid = (max(1, ct.cdiv(T, tile_m)), B * self.n_head, 1)
        
        ct.launch(torch.cuda.current_stream(), grid, microgpt_attention_kernel,
                 (q, k, v, out, scale_log2, head_dim, self.n_head, tile_m, 16, 2, 5))
        
        attn_out = out.transpose(1, 2).reshape(B, T, -1)
        x = x + self.wo(attn_out)
        
        # MLP path
        x_norm = run_rmsnorm_op(x, self.ln2)
        mlp_h = torch.relu(self.fc1(x_norm))
        x = x + self.fc2(mlp_h)
        return x

class MicroGPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.Parameter(torch.ones(n_embd))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))
        x = self.blocks(x)
        x = run_rmsnorm_op(x, self.ln_f)
        return self.lm_head(x)

if __name__ == "__main__":
    device = 'cuda'
    model = MicroGPT(vocab_size=64, n_embd=128, n_head=4, n_layer=2, block_size=64).to(device).half()
    idx = torch.randint(0, 64, (2, 64), device=device)
    logits = model(idx)
    print(f"Success! Logits shape: {logits.shape}")
