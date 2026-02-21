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
        
        h = x0
        loops = num_loops if num_loops is not None else self.num_loops
        
        # 1. Determine Padded Shapes (M and N/V)
        tile_size_m = 4
        M_padded = math.ceil(M / tile_size_m) * tile_size_m
        N = self.config.n_embd
        V = self.config.vocab_size
        tile_n = next_pow2(N)
        tile_v = next_pow2(V)

        # 2. Initialize State Tensors with M-padding
        active_mask = torch.ones((M_padded, 1), device=device, dtype=torch.float32)
        steps_taken = torch.zeros((M_padded, 1), device=device, dtype=torch.int32)
        
        for l in range(loops):
            if not self.training and halt_threshold is not None:
                if not (active_mask[:M] > 0.5).any(): 
                    break
                
            h_input = h + x0 
            step_enc = self.step_embedding(torch.tensor([l], device=device))
            h_input = h_input + step_enc
            
            h_next = self.transformer.h(h_input)
            
            if halt_threshold is not None and not self.training:
                with torch.no_grad():
                    # Flatten and M-Pad states
                    h_flat = h.view(M, N)
                    h_next_flat = h_next.view(M, N)
                    h_padded = F.pad(h_flat, (0, tile_n - N, 0, M_padded - M))
                    h_next_padded = F.pad(h_next_flat, (0, tile_n - N, 0, M_padded - M))
                    
                    # Compute and pad logits
                    logits_step = self.lm_head(self.transformer.ln_f(h_next))
                    logits_flat = logits_step.view(M, V)
                    logits_padded = F.pad(logits_flat, (0, tile_v - V, 0, M_padded - M), value=-float('inf'))
                    
                    # 3. Launch cuTile Kernel
                    grid = (M_padded // tile_size_m,)
                    ct.launch(torch.cuda.current_stream(), grid, looplm_halt_update_kernel,
                             (h_padded, h_next_padded, logits_padded, active_mask, steps_taken, 
                              halt_threshold, tile_size_m, tile_n, tile_v))
                    
                    # 4. Unpad and Update
                    h = h_padded[:M, :N].view(b, t, -1)
            else:
                h = h_next
                steps_taken[:M] += active_mask[:M].int()
            
        h = self.transformer.ln_f(h)
        if targets is not None:
            logits = self.lm_head(h)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(h[:, [-1], :])
            loss = None

        return logits, loss, steps_taken[:M].view(b, t)
