---
title: "NanoGPT System v1 – Standard GPT-2 Tiled Transition"
source: "nanoGPT/model.py + FMHAv4 Compound Knowledge"
extraction-date: 2026-02-13
tags: [NanoGPT, GPT-2, cuTile, WeightTying, CompoundEngineering]
status: "active"
---

system NanoGPT_System_v1 {

    // ============================================================
    // 1. Design Space (High-Level Archetype)
    // ============================================================
    design_space {
        attention_engine: ["cutile_fmha_v4", "pytorch_sdpa"]
        normalization: ["layernorm_with_bias", "rmsnorm"] // nanoGPT uses LN
        activation: ["gelu_new", "relu"] 
        weight_strategy: ["tied_embedding_head", "separate"]
        positional_encoding: ["learned_absolute", "rope"]
    }

    // ============================================================
    // 2. Tuning Space (Performance Axes)
    // ============================================================
    tuning_space {
        // Inherited from FMHAv4 verified RTX 5070 results
        tile_m: [64] 
        tile_n: [64, 128]
        tile_d: [64, 128] // head_dim is 64 for GPT-2 Small
        
        // Blackwell TMA Load Latencies
        k_lat: [2, 3]
        v_lat: [4, 5]
    }

    // ============================================================
    // 3. Model & Knowledge (Semantic Intelligence)
    // ============================================================
    model {
        type GPT2Config matches { 
            n_layer: 12, n_embd: 768, n_head: 12, block_size: 1024, vocab_size: 50304
        }
        state current_design: "tied_flash_layernorm"
    }

    knowledge {
        // --- Inherited from FMHAv4 (Blackwell Design Laws) ---
        fact rtx5070_fmha_peak {
            description: "On RTX 5070, attention achieves >130 TFLOPS with 64x64 tiles."
            inherited_from: "fmha_system_v4.dsl"
            confidence: 1.0
        }

        fact tma_pipelining_law {
            description: "Causal workloads favor V_Lat=5 for memory overlapping."
            inherited_from: "last_engineering_trace.json"
        }

        // --- Invariants from nanoGPT/model.py ---
        invariant WeightTying {
            assert: "model.transformer.wte.weight == model.lm_head.weight"
            source: "model.py:L130"
        }

        invariant ResidualScaling {
            assert: "Initial normal_(std=0.02 / sqrt(2 * n_layer)) for c_proj weights"
            source: "model.py:L135"
        }

        // --- Optimization Rules ---
        rule "Vectorize Attention" {
            when: "attention_engine == 'cutile_fmha_v4'"
            apply: "Mapping (B, T, nh, hs) -> (B, nh, T, hs) for optimized Tiled Load."
        }

        rule "LayerNorm Bias Handling" {
            when: "normalization == 'layernorm_with_bias'"
            apply: "Use standard cuTile LN with additive bias in final store."
        }
    }

    // ============================================================
    // 4. Trace Schema
    // ============================================================
    trace_schema {
        variant TraceItem {
            case Performance { tflops: float, speedup_vs_native: float }
            case Memory { utilization: float, bandwidth_gbs: float }
            case Correctness { max_diff: float, passed: bool }
        }
    }

    // ============================================================
    // 5. Loops
    // ============================================================

    agent_loop NanoGPT_Implementer {
        step "Map Module to Kernels" {
            llm.query {
                prompt: "Map nanoGPT/model.py modules to cuTile kernels using design_space choices."
                output_var: "implementation_plan"
            }
        }

        step "Generate Implementation" {
            tool.write {
                path: "nanogpt_cutile.py"
                content: "/* Implement GPT class using compiled cuTile blocks */"
            }
        }

        step "Benchmark vs Native" {
            tool.run { cmd: "python bench_nanogpt.py --backend cutile --compare_to native" }
        }
    }

    // ============================================================
    // 6. Build
    // ============================================================
    build {
        artifact "nanogpt_v1_report.md"
        artifact "nanogpt_cutile_implementation.py"
    }
}
