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
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x0 = self.transformer.drop(tok_emb + pos_emb)
        
        h = x0
        loops = num_loops if num_loops is not None else self.num_loops
        
        # State tensors for kernel management
        active_mask = torch.ones((b * t, 1), device=device, dtype=torch.float32)
        steps_taken = torch.zeros((b * t, 1), device=device, dtype=torch.int32)
        
        for l in range(loops):
            if not self.training and halt_threshold is not None:
                if not (active_mask > 0.5).any(): 
                    break
                
            # Rule: Inject X0 Residue + Step Encoding
            h_input = h + x0 
            step_enc = self.step_embedding(torch.tensor([l], device=device))
            h_input = h_input + step_enc
            
            # Compute next state
            h_next = self.transformer.h(h_input)
            
            # Phase 2: cuTile Halt Update Kernel (Inference only)
            if halt_threshold is not None and not self.training:
                with torch.no_grad():
                    # 1. Determine Power-of-Two Shapes
                    N = self.config.n_embd
                    V = self.config.vocab_size
                    tile_n = next_pow2(N)
                    tile_v = next_pow2(V)
                    
                    # 2. Flatten and Pad
                    h_flat = h.view(-1, N)
                    h_next_flat = h_next.view(-1, N)
                    
                    # Compute logits temporarily for halting decision
                    logits_step = self.lm_head(self.transformer.ln_f(h_next))
                    logits_flat = logits_step.view(-1, V)
                    
                    # Pad to Power-of-Two
                    h_padded = F.pad(h_flat, (0, tile_n - N))
                    h_next_padded = F.pad(h_next_flat, (0, tile_n - N))
                    # Pad logits with -inf so max() correctly identifies valid confidence
                    logits_padded = F.pad(logits_flat, (0, tile_v - V), value=-float('inf'))
                    
                    # 3. Launch cuTile Kernel for Atomic Halt & Update
                    grid = ( (b * t + 3) // 4, ) # TILE_M=4
                    ct.launch(torch.cuda.current_stream(), grid, looplm_halt_update_kernel,
                             (h_padded, h_next_padded, logits_padded, active_mask, steps_taken, 
                              halt_threshold, 4, tile_n, tile_v))
                    
                    # 4. Unpad and Update
                    # The kernel updates h_padded in-place (H_current)
                    h = h_padded[:, :N].view(b, t, -1)
            else:
                h = h_next
                steps_taken += active_mask.int()
            
        h = self.transformer.ln_f(h)
        if targets is not None:
            logits = self.lm_head(h)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(h[:, [-1], :])
            loss = None

        return logits, loss, steps_taken.view(b, t)
