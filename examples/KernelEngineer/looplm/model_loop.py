import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import cuda.tile as ct

# Reuse shared logic
from model import Block, LayerNorm, GPTConfig
from looplm_kernels import looplm_halt_update_kernel

def next_pow2(n):
    return 2**(n - 1).bit_length()

class LoopGPT(nn.Module):
    def __init__(self, config, num_loops=12, inject_x0=True):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.num_loops = num_loops
        self.inject_x0 = inject_x0 # Phase 4.2: Toggle for Anchor Injection

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = Block(config), 
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight 
        self.step_embedding = nn.Embedding(num_loops, config.n_embd)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * 1))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def _get_cos_sin(self, T, device, dtype):
        head_dim = self.config.n_embd // self.config.n_head
        inv_freq = self.transformer.h.attn.inv_freq
        t = torch.arange(T, device=device, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)
        cos = freqs.cos().view(1, 1, T, head_dim).to(dtype)
        sin = freqs.sin().view(1, 1, T, head_dim).to(dtype)
        return cos, sin

    def forward(self, idx, targets=None, num_loops=None, halt_threshold=None, thinking_token_id=None, thinking_threshold=None):
        device = idx.device
        b, t = idx.size()
        M = b * t
        
        tok_emb = self.transformer.wte(idx)
        x0 = self.transformer.drop(tok_emb) # No more pos_emb
        cos, sin = self._get_cos_sin(t, device, tok_emb.dtype)
        
        h = x0
        loops = num_loops if num_loops is not None else self.num_loops
        N = self.config.n_embd
        V = self.config.vocab_size

        # Wait-to-Think: detect if '=' has appeared in the sequence
        # We assume thinking_token_id is passed (e.g., stoi['='])
        is_thinking = torch.zeros((b, t), device=device, dtype=torch.float32)
        if thinking_token_id is not None:
            # For each position, mark 1 if thinking_token_id appeared at or before this position
            for i in range(b):
                eq_indices = (idx[i] == thinking_token_id).nonzero(as_tuple=True)[0]
                if len(eq_indices) > 0:
                    first_eq = eq_indices[0].item()
                    is_thinking[i, first_eq:] = 1.0

        # --- Training Mode (Refined for Reasoning) ---
        if self.training:
            h_curr = h
            supervised_logits = []
            for l in range(loops):
                # Phase 4.2: Anchor Injection (h = Block(h + x0) vs Block(h))
                h_input = h_curr + x0 if self.inject_x0 else h_curr
                step_enc = self.step_embedding(torch.tensor([l], device=device))
                h_next = self.transformer.h(h_input + step_enc, cos=cos, sin=sin)
                h_curr = h_next
                
                # Supervise second half for stability
                if l >= loops // 2:
                    lg = self.lm_head(self.transformer.ln_f(h_next))
                    supervised_logits.append(lg)
            
            if targets is not None:
                losses = []
                for lg in supervised_logits:
                    l = F.cross_entropy(lg.view(-1, V), targets.view(-1), ignore_index=-1)
                    losses.append(l)
                
                loss = torch.stack(losses).mean()
                logits = supervised_logits[-1]
            else:
                logits = self.lm_head(self.transformer.ln_f(h_curr))
                loss = None
            
            return logits, loss, torch.zeros((b, t), device=device, dtype=torch.int32)

        # --- Inference Mode (Pinned-State cuTile) ---
        else:
            tile_size_m = 4
            M_padded = math.ceil(M / tile_size_m) * tile_size_m
            tile_n = next_pow2(N)
            tile_v = next_pow2(V)

            h_state_padded = torch.zeros((M_padded, tile_n), device=device, dtype=x0.dtype)
            h_state_padded[:M, :N] = x0.view(M, N)
            active_mask = torch.zeros((M_padded, 1), device=device, dtype=torch.float32)
            active_mask[:M, 0] = 1.0 
            steps_taken = torch.zeros((M_padded, 1), device=device, dtype=torch.int32)
            
            # Per-token halt threshold
            token_thresholds = torch.full((M_padded, 1), halt_threshold if halt_threshold is not None else 0.0, device=device, dtype=torch.float32)
            if thinking_threshold is not None and thinking_token_id is not None:
                token_thresholds[:M, 0] = torch.where(is_thinking.view(M) > 0.5, 
                                                      torch.tensor(thinking_threshold, device=device), 
                                                      torch.tensor(halt_threshold if halt_threshold is not None else 0.0, device=device))

            x0_padded = torch.zeros((M_padded, tile_n), device=device, dtype=x0.dtype)
            x0_padded[:M, :N] = x0.view(M, N)

            h_next_padded = torch.zeros((M_padded, tile_n), device=device, dtype=x0.dtype)
            logits_padded = torch.full((M_padded, tile_v), -float('inf'), device=device, dtype=x0.dtype)

            for l in range(loops):
                if halt_threshold is not None or thinking_threshold is not None:
                    if not (active_mask[:M] > 0.5).any(): break
                
                h_current_view = h_state_padded[:M, :N].view(b, t, N)
                x0_current_view = x0_padded[:M, :N].view(b, t, N)
                
                # Phase 4.2: Anchor Injection (h = Block(h + x0) vs Block(h))
                h_input = h_current_view + x0_current_view if self.inject_x0 else h_current_view
                step_enc = self.step_embedding(torch.tensor([l], device=device))
                h_next = self.transformer.h(h_input + step_enc, cos=cos, sin=sin)
                
                if halt_threshold is not None or thinking_threshold is not None:
                    with torch.no_grad():
                        # Phase 4.1: Hybrid Inference (Attention + MLP in PyTorch, Halt in Kernel)
                        # This ensures 100% numerical parity with Training path.
                        
                        # Full Transformer Block in PyTorch
                        h_input = h_current_view + x0_current_view if self.inject_x0 else h_current_view
                        step_enc = self.step_embedding(torch.tensor([l], device=device))
                        h_next = self.transformer.h(h_input + step_enc)
                        
                        # Prepare buffers for Halting Kernel
                        h_next_padded.zero_()
                        logits_padded.fill_(-float('inf'))
                        
                        logits_step = self.lm_head(self.transformer.ln_f(h_next))
                        logits_padded[:M, :V] = logits_step.view(M, V)
                        h_next_padded[:M, :N] = h_next.view(M, N)
                        
                        grid_x = M_padded // tile_size_m
                        ct.launch(torch.cuda.current_stream(), (grid_x,), looplm_halt_update_kernel,
                                 (h_state_padded, h_next_padded, logits_padded, active_mask, steps_taken, 
                                  token_thresholds, tile_size_m, tile_n, tile_v, V))
                else:
                    h_state_padded[:M, :N] = h_next.view(M, N)
                    steps_taken[:M] += 1
            
            h_final = h_state_padded[:M, :N].view(b, t, N)
            h_final = self.transformer.ln_f(h_final)
            
            if targets is not None:
                logits = self.lm_head(h_final)
                loss = F.cross_entropy(logits.view(-1, V), targets.view(-1), ignore_index=-1)
            else:
                logits = self.lm_head(h_final[:, [-1], :])
                loss = None
            
            return logits, loss, steps_taken[:M].view(b, t)
