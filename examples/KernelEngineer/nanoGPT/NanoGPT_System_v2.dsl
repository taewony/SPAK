---
title: "NanoGPT System v2 – Production Grade Architecture"
source: "nanoGPT/model.py + TileGym/ops + SPAK Compound Knowledge"
extraction-date: 2026-02-13
tags: [NanoGPT, GPT-2, TileGym, Blackwell, SystematicEngineering]
status: "active"
---

system NanoGPT_System_v2 {

    // ============================================================
    // 1. Design Space (Architectural Specification)
    // ============================================================
    design_space {
        attention: {
            engine: ["TileGym.fmha", "Inlined_FMHA_v4"]
            causal: true
            scaling: "1/sqrt(head_dim)"
        }
        normalization: {
            type: ["LayerNorm", "RMSNorm"]
            bias: [true, false]
            implementation: ["Strategic_Inlined", "TileGym.persistent_layer_norm"]
        }
        mlp: {
            activation: ["gelu_tanh"] // GPT-2 standard
            fused: [true, false]
        }
        state_management: {
            kv_cache: [true, false]
            weight_tying: true
        }
    }

    // ============================================================
    // 2. Tuning Space (Blackwell Optimized)
    // ============================================================
    tuning_space {
        // Core Tiling Heuristics (Verified on RTX 5070)
        tile_m: [64] 
        tile_n: [64, 128]
        tile_d: [64] 
        
        // Normalization Tiling (Stability choice)
        ln_tile_m: [4] // Found optimal for row-wise reduction stability
        
        // Memory Pipelining
        k_lat: [2, 3]
        v_lat: [4, 5]
        
        // SMS Utilization
        num_sms: 80 
    }

    // ============================================================
    // 3. Knowledge & Invariants (The Cognitive Layer)
    // ============================================================
    model {
        type GPT2_Small { 
            L: 12, D: 768, H: 12, T: 1024, V: 50304 
        }
    }

    knowledge {
        // --- Verified Facts (Production Scaling Result) ---
        fact nanogpt_cutile_performance {
            description: "On RTX 5070, achieved 2.64x speedup vs. native PyTorch SDPA/LN baseline."
            avg_step_time: "2.8ms"
            baseline_time: "7.4ms"
            confidence: 1.0
            source: "cutile.log comparison"
        }

        fact layernorm_masking_rule {
            description: "LayerNorm for non-pow2 dims (384) must use pow2 tiling (512) + masked variance sum + TILE_M=4."
            implementation: "Verified in nanogpt_cutile.py"
            stability: 1.0
        }

        fact gelu_tanh_invariant {
            description: "GELU activation must use the Tanh approximation to match GPT-2 weights exactly."
            source: "model.py:L81"
        }

        // --- Inherited Stability floor (From MicroGPT) ---
        fact numerical_stability_floor {
            description: "Use SAFE_NEG_VAL = -1e20 for attention masks to prevent NaN in half precision."
            confidence: 1.0
        }

        // --- Inherited Performance facts (From FMHAv4) ---
        fact blackwell_tma_optimal {
            description: "RTX 5070 attention peaks at 135 TFLOPS with V_Lat=5 and 64x64 tiles."
            confidence: 1.0
        }

        // --- GPT-2 Specific Rules (From model.py) ---
        invariant WeightTying {
            assert: "model.transformer.wte.weight == model.lm_head.weight"
        }

        rule "Residual Scaling Heuristic" {
            when: "layer_index > 0"
            apply: "Initialize c_proj weights with std = 0.02 / sqrt(2 * n_layer)"
            source: "model.py:L135"
        }

        rule "Persistent LN Optimization" {
            when: "sequence_length > num_sms * 2"
            apply: "Select TileGym.persistent_layer_norm"
            reason: "Better wave utilization on Blackwell."
        }
    }

    // ============================================================
    // 4. Operational Loops
    // ============================================================

    agent_loop NanoGPT_Orchestrator {
        step "Assemble Modular Backend" {
            llm.query {
                prompt: "Map NanoGPT blocks to TileGym dispatchers using design_space."
                output_var: "module_map"
            }
        }

        step "Verify Parameter Alignment" {
            tool.run { 
                cmd: "python nanogpt_cutile.py --check_params" 
            }
        }

        step "Distributed Performance Sweep" {
            tool.run { 
                cmd: "python bench_nanogpt.py --configs tuning_space" 
            }
        }
    }

    // ============================================================
    // 5. Build Artifacts
    // ============================================================
    build {
        artifact "nanogpt_production_implementation.py" {
            base: "nanoGPT/model.py"
            transforms: ["Vectorize", "Tiling", "TileGym-Interop"]
        }
        artifact "performance_fidelity_report.md"
    }
}
