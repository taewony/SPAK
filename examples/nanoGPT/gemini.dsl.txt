system "NanoGPT_TileGym_Implementation" {
    version: "1.0";
    target: "NVIDIA RTX 5070";
    backend: "cuTile";
    base_repo: "karpathy/nanoGPT";
    
    /* Global Reference to Hardware Truths */
    import "kernels/fmha_blackwell.dsl" as FMHA_Spec;

    knowledge "Model_Hyperparameters" {
        // NanoGPT config matching 'gpt2' (124M)
        n_layer: 12;
        n_head: 12;
        n_embd: 768;
        block_size: 1024; // Context Length
        vocab_size: 50257;
        dropout: 0.0; // Optimized out for inference
        bias: false;  // Bias-less for better GEMM performance on Tensor Cores
    }

    /* 1. Low-level Operator Abstraction (TileGym Mapping) */
    module "TileOps" {
        type: "interface";
        description: "Mapping to NVIDIA TileGym primitives";

        op "Linear" {
            source: "tilegym.ops.cutile.linear";
            backend_func: "tile_linear";
            invariant: "y = x @ W.T + b";
        }

        op "LayerNorm" {
            source: "tilegym.ops.cutile.layernorm";
            backend_func: "tile_layernorm";
            invariant: "y = (x - mean) / sqrt(var + eps) * weight + bias";
        }

        op "GELU" {
            source: "tilegym.ops.cutile.activation";
            backend_func: "tile_gelu";
            approximation: "tanh"; // NewGELU
        }

        op "FlashAttention" {
            reference: FMHA_Spec; // Previously defined SNF
            constraint: "Must support causal masking";
        }
    }

    /* 2. Component Module: MLP (Feed-Forward Network) */
    module "MLP" {
        type: "layer";
        input: "x (B, T, C)";
        output: "y (B, T, C)";

        knowledge "Architecture" {
            expansion_factor: 4;
            inner_dim: "n_embd * 4";
        }

        trace "Forward_Trace" {
            step_1: "c_fc = TileOps.Linear(x, in=n_embd, out=inner_dim)";
            step_2: "hidden = TileOps.GELU(c_fc)";
            step_3: "c_proj = TileOps.Linear(hidden, in=inner_dim, out=n_embd)";
            return: "c_proj";
        }
        
        invariant "Dimension_Preservation" {
            assert: "input.shape == output.shape";
        }
    }

    /* 3. Component Module: Block (Transformer Layer) */
    module "Block" {
        type: "layer";
        description: "Pre-LayerNorm Transformer Block";

        components {
            ln_1: "TileOps.LayerNorm(n_embd)";
            attn: "TileOps.FlashAttention(config)";
            ln_2: "TileOps.LayerNorm(n_embd)";
            mlp: "Module.MLP(config)";
        }

        rule "Residual_Connection" {
            logic: "x = x + SubModule(Norm(x))";
            reason: "Mitigates vanishing gradient in deep networks";
        }

        trace "Execution_Flow" {
            /* Attention Path */
            1: "norm_1 = ln_1(x)";
            2: "attn_out = attn(norm_1)";
            3: "x = x + attn_out"; // Residual 1

            /* MLP Path */
            4: "norm_2 = ln_2(x)";
            5: "mlp_out = mlp(norm_2)";
            6: "x = x + mlp_out"; // Residual 2
        }
    }

    /* 4. System Module: GPT (Full Model) */
    module "GPT" {
        type: "model";
        description: "Decoder-only Transformer";

        components {
            wte: "Embedding(vocab_size, n_embd)"; // Word Token Embedding
            wpe: "Embedding(block_size, n_embd)"; // Word Pos Embedding
            blocks: "List[Module.Block] * n_layer";
            ln_f: "TileOps.LayerNorm(n_embd)";
            lm_head: "TileOps.Linear(n_embd, vocab_size, bias=False)";
        }

        invariant "Weight_Tying" {
            constraint: "wte.weight == lm_head.weight";
            reason: "Significantly reduces parameter count";
        }

        trace "Generate_Next_Token" {
            input: "idx (B, T)";
            
            1: "tok_emb = wte(idx)";
            2: "pos_emb = wpe(arange(T))";
            3: "x = tok_emb + pos_emb";
            
            /* Stack Execution */
            4: "for block in blocks: x = block(x)";
            
            /* Final Projection */
            5: "x = ln_f(x)";
            6: "logits = lm_head(x)"; // (B, T, vocab_size)
            
            /* Inference Optimization */
            7: "next_token_logits = logits[:, -1, :]";
        }
        
        rule "Weight_Initialization" {
            // NanoGPT specific init
            target: "Linear, Embedding";
            strategy: "Normal(mean=0.0, std=0.02)";
            exception: "Residual projections (c_proj) scale by 1/sqrt(2 * n_layer)";
        }
    }

    /* 5. Experiment & Verification */
    experiment "Convergence_Check" {
        hypothesis: "TileGym implementation matches PyTorch reference loss within epsilon";
        metric: "CrossEntropyLoss";
        threshold: "1e-4";
        dataset: "Shakespeare_char"; // Simple dataset for rapid verification
    }
}