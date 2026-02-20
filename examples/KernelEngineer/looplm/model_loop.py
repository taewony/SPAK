import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

# We will reuse the same block but apply it iteratively
from model import Block, LayerNorm, GPTConfig

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
            # Only ONE block for LoopLM
            h = Block(config), 
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight 

        # Positional Encoding for Thinking Steps (Optional for v1, but good to have)
        # For v1, we start with standard RoPE or just the same PE.
        # Let's add a simple Thinking Step Embedding as per v1 dsl
        self.step_embedding = nn.Embedding(num_loops, config.n_embd)

        self.apply(self._init_weights)
        # apply special scaled init to the residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * 1)) # L=1 here

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, num_loops=None):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x0 = self.transformer.drop(tok_emb + pos_emb)
        
        h = x0
        loops = num_loops if num_loops is not None else self.num_loops
        
        for l in range(loops):
            # Rule: Inject X0 Residue (h = h + LN(x0))
            # Note: In some versions it's h = h + Block(LN(h + LN(x0)))
            # Let's follow the plan.md: h_{t+1} = h_t + F(LN(h_t + LN(x0)))
            
            # For simplicity in v1, let's do:
            h_input = h + x0 # Simple injection
            
            # Thinking Step Encoding
            step_enc = self.step_embedding(torch.tensor([l], device=device))
            h_input = h_input + step_enc
            
            h = self.transformer.h(h_input)
            
        h = self.transformer.ln_f(h)
        
        if targets is not None:
            logits = self.lm_head(h)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(h[:, [-1], :])
            loss = None

        return logits, loss
