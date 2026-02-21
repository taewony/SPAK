import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import cuda.tile as ct

# Reuse shared logic from standard model
from model import Block, LayerNorm, GPTConfig
from looplm_kernels import looplm_halt_update_kernel

def next_pow2(n):
    return 2**(n - 1).bit_length()

class LoopGPT(nn.Module):
    def __init__(self, config, num_loops=12):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.num_loops = num_loops

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
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
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, num_loops=None, halt_threshold=None):
        device = idx.device
        b, t = idx.size()
        M = b * t
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x0 = self.transformer.drop(tok_emb + pos_emb)
        
        # 1. Pinned State Setup (Architecture-level Fix)
        tile_size_m = 4
        M_padded = math.ceil(M / tile_size_m) * tile_size_m
        N = self.config.n_embd
        V = self.config.vocab_size
        tile_n = next_pow2(N)
        tile_v = next_pow2(V)

        # Main State Buffers
        h_state_padded = torch.zeros((M_padded, tile_n), device=device, dtype=x0.dtype)
        h_state_padded[:M, :N] = x0.view(M, N)
        active_mask = torch.zeros((M_padded, 1), device=device, dtype=torch.float32)
        active_mask[:M, 0] = 1.0 
        steps_taken = torch.zeros((M_padded, 1), device=device, dtype=torch.int32)
        
        # Persistent X0 for Injection
        x0_padded = torch.zeros((M_padded, tile_n), device=device, dtype=x0.dtype)
        x0_padded[:M, :N] = x0.view(M, N)

        # Auxiliary Pinned Buffers
        h_next_padded = torch.zeros((M_padded, tile_n), device=device, dtype=x0.dtype)
        logits_padded = torch.full((M_padded, tile_v), -float('inf'), device=device, dtype=x0.dtype)
        
        loops = num_loops if num_loops is not None else self.num_loops
        all_logits = [] 
        
        for l in range(loops):
            if not self.training and halt_threshold is not None:
                if not (active_mask[:M] > 0.5).any(): break
            
            h_current = h_state_padded[:M, :N].view(b, t, N)
            x0_current = x0_padded[:M, :N].view(b, t, N)
            
            # Update Step
            h_input = h_current + x0_current
            step_enc = self.step_embedding(torch.tensor([l], device=device))
            h_input = h_input + step_enc
            
            h_next = self.transformer.h(h_input)
            h_next_flat = h_next.view(M, N)
            
            if not self.training and halt_threshold is not None:
                with torch.no_grad():
                    h_next_padded.zero_()
                    logits_padded.fill_(-float('inf'))
                    
                    logits_step = self.lm_head(self.transformer.ln_f(h_next))
                    logits_padded[:M, :V] = logits_step.view(M, V)
                    h_next_padded[:M, :N] = h_next_flat
                    
                    # Launch Kernel with explicit M_padded grid
                    grid_x = M_padded // tile_size_m
                    ct.launch(torch.cuda.current_stream(), (grid_x,), looplm_halt_update_kernel,
                             (h_state_padded, h_next_padded, logits_padded, active_mask, steps_taken, 
                              halt_threshold, tile_size_m, tile_n, tile_v))
            else:
                # Update logic for Training or Fixed-loop inference
                # During training, we bypass the halting kernel to maintain Autograd graph
                h_state_padded[:M, :N] = h_next_flat
                steps_taken[:M] += 1
                
                if self.training:
                    logits_l = self.lm_head(self.transformer.ln_f(h_next))
                    all_logits.append(logits_l)
            
        # Final Readout
        h_final = h_state_padded[:M, :N].view(b, t, N)
        h_final = self.transformer.ln_f(h_final)
        
        if targets is not None:
            if len(all_logits) > 0:
                # Multi-step Supervision: loss across all iterations
                losses = [F.cross_entropy(lg.view(-1, V), targets.view(-1), ignore_index=-1) for lg in all_logits]
                loss = torch.stack(losses).mean()
            else:
                logits = self.lm_head(h_final)
                loss = F.cross_entropy(logits.view(-1, V), targets.view(-1), ignore_index=-1)
            logits = self.lm_head(h_final) 
        else:
            logits = self.lm_head(h_final[:, [-1], :])
            loss = None

        return logits, loss, steps_taken[:M].view(b, t)
