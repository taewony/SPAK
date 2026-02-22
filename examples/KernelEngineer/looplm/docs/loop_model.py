# =============================================================================
# LoopGPT Model: Semantic Norm / Pseudo Code Representation
# =============================================================================
#
# This file reorganizes the logic of model_loop.py into a clear, readable
# pseudo‑code format, preserving all function and class names.
# It highlights the two execution modes (training and inference) and the
# key data flows.
# =============================================================================

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import cuda.tile as ct

from model import Block, LayerNorm, GPTConfig
from looplm_kernels import looplm_halt_update_kernel

# -----------------------------------------------------------------------------
# Helper: next power of two (for tile size alignment)
# -----------------------------------------------------------------------------
def next_pow2(n):
    return 2 ** ((n - 1).bit_length())

# -----------------------------------------------------------------------------
# Main Model Class
# -----------------------------------------------------------------------------
class LoopGPT(nn.Module):
    def __init__(self, config, num_loops=12):
        super().__init__()
        self.config = config
        self.num_loops = num_loops

        # Transformer core (single block, shared weights, tied embeddings)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),   # token embeddings
            "wpe": nn.Embedding(config.block_size, config.n_embd),   # position embeddings
            "drop": nn.Dropout(config.dropout),
            "h": Block(config),                                      # one transformer block
            "ln_f": LayerNorm(config.n_embd, bias=config.bias),      # final layernorm
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight            # weight tying

        # Step embedding (distinguishes each loop iteration)
        self.step_embedding = nn.Embedding(num_loops, config.n_embd)

        self.apply(self._init_weights)
        # Special initialisation for the projection layers (c_proj)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * 1))

    def _init_weights(self, module):
        # standard weight init
        ...

    # -------------------------------------------------------------------------
    # Forward Pass
    # -------------------------------------------------------------------------
    def forward(self, idx, targets=None, num_loops=None, halt_threshold=None):
        device = idx.device
        b, t = idx.size()                     # batch, time
        M = b * t                              # total number of tokens
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # Initial embeddings (x0)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x0 = self.transformer.drop(tok_emb + pos_emb)   # shape (b, t, n_embd)

        loops = num_loops if num_loops is not None else self.num_loops
        N = self.config.n_embd
        V = self.config.vocab_size

        # ==================== TRAINING MODE ====================
        if self.training:
            # We use full autograd and half‑loop supervision.
            h_curr = x0
            supervised_logits = []

            for l in range(loops):
                # Prepare input: persistent x0 injection + step embedding
                h_input = h_curr + x0
                step_enc = self.step_embedding(torch.tensor([l], device=device))
                h_next = self.transformer.h(h_input + step_enc)

                h_curr = h_next

                # Supervise only the second half of the loop (for stability)
                if l >= loops // 2:
                    lg = self.lm_head(self.transformer.ln_f(h_next))
                    supervised_logits.append(lg)

            if targets is not None:
                losses = [F.cross_entropy(lg.view(-1, V), targets.view(-1), ignore_index=-1)
                          for lg in supervised_logits]
                loss = torch.stack(losses).mean()
                logits = supervised_logits[-1]
            else:
                logits = self.lm_head(self.transformer.ln_f(h_curr))
                loss = None

            # steps_taken is meaningless during training (return zeros)
            return logits, loss, torch.zeros((b, t), device=device, dtype=torch.int32)

        # ==================== INFERENCE MODE ====================
        else:
            # Use pinned state tensors and the cuTile halting kernel.
            # Tile sizes (must be powers of two for the kernel)
            tile_size_m = 4
            M_padded = math.ceil(M / tile_size_m) * tile_size_m
            tile_n = next_pow2(N)
            tile_v = next_pow2(V)

            # ---- Pinned buffers (created once, reused in the loop) ----
            # Hidden state (padded to tile_n)
            h_state_padded = torch.zeros((M_padded, tile_n), device=device, dtype=x0.dtype)
            h_state_padded[:M, :N] = x0.view(M, N)

            # Active mask (1 = token still thinking, 0 = halted)
            active_mask = torch.zeros((M_padded, 1), device=device, dtype=torch.float32)
            active_mask[:M, 0] = 1.0

            # Steps taken counter
            steps_taken = torch.zeros((M_padded, 1), device=device, dtype=torch.int32)

            # Padded copy of x0 (used in each loop)
            x0_padded = torch.zeros((M_padded, tile_n), device=device, dtype=x0.dtype)
            x0_padded[:M, :N] = x0.view(M, N)

            # Temporary buffers for the next state and logits (reused)
            h_next_padded = torch.zeros((M_padded, tile_n), device=device, dtype=x0.dtype)
            logits_padded = torch.full((M_padded, tile_v), -float('inf'), device=device, dtype=x0.dtype)

            # ---- Iterate over loops (adaptive halting) ----
            for l in range(loops):
                # Early exit if no active tokens remain
                if halt_threshold is not None:
                    if not (active_mask[:M] > 0.5).any():
                        break

                # Prepare input for the shared block (using current padded state)
                h_current_view = h_state_padded[:M, :N].view(b, t, N)
                x0_current_view = x0_padded[:M, :N].view(b, t, N)

                h_input = h_current_view + x0_current_view
                step_enc = self.step_embedding(torch.tensor([l], device=device))
                h_next = self.transformer.h(h_input + step_enc)          # (b, t, N)

                if halt_threshold is not None:
                    with torch.no_grad():
                        # Reset temporary buffers
                        h_next_padded.zero_()
                        logits_padded.fill_(-float('inf'))

                        # Compute logits for halting decision
                        logits_step = self.lm_head(self.transformer.ln_f(h_next))
                        logits_padded[:M, :V] = logits_step.view(M, V)
                        h_next_padded[:M, :N] = h_next.view(M, N)

                        # Launch the halting kernel (updates h_state_padded, active_mask, steps_taken in‑place)
                        grid_x = M_padded // tile_size_m
                        ct.launch(
                            torch.cuda.current_stream(),
                            (grid_x,),
                            looplm_halt_update_kernel,
                            (h_state_padded, h_next_padded, logits_padded,
                             active_mask, steps_taken,
                             halt_threshold, tile_size_m, tile_n, tile_v, V)
                        )
                else:
                    # No halting: simply update state and increment steps for all real tokens
                    h_state_padded[:M, :N] = h_next.view(M, N)
                    steps_taken[:M] += 1

            # ---- Final readout ----
            h_final = h_state_padded[:M, :N].view(b, t, N)
            h_final = self.transformer.ln_f(h_final)

            if targets is not None:
                logits = self.lm_head(h_final)
                loss = F.cross_entropy(logits.view(-1, V), targets.view(-1), ignore_index=-1)
            else:
                logits = self.lm_head(h_final[:, [-1], :])   # last token only for generation
                loss = None

            return logits, loss, steps_taken[:M].view(b, t)
