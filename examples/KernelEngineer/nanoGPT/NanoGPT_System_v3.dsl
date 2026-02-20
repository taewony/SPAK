---
title: "NanoGPT System v3 – Verified Systematic Engineering"
source: "nanoGPT/model.py + cuTile Regression + Autograd Lessons"
extraction-date: 2026-02-19
tags: [NanoGPT, Autograd, Parity, Blackwell, CompoundEngineering]
status: "active"
---

system NanoGPT_System_v3 {

    // ============================================================
    // 1. Design Space (Architectural Specification)
    // ============================================================
    design_space {
        execution_mode: ["Hybrid_Autograd", "Full_cuTile_Bwd"]
        attention: {
            engine: ["TileGym.fmha", "Inlined_FMHA_v4"]
            causal: true
            stability: "Finite_Neg_Inf" // -1e20 law
        }
        normalization: {
            type: ["LayerNorm", "RMSNorm"]
            implementation: ["Contiguous_Padded_cuTile", "PyTorch_Native"]
        }
        validation_protocol: ["Hierarchical_Parity_L1_L3"]
    }

    // ============================================================
    // 2. Tuning Space (Blackwell Optimized)
    // ============================================================
    tuning_space {
        tile_m: [64] 
        tile_n: [64]
        k_lat: [2]
        v_lat: [5]
        
        // New: Memory Safety Invariants
        memory_alignment: "Element_Offset_Ambiguity_Resolved"
        indexing_mode: "Tile_Based_Units" 
    }

    // ============================================================
    // 3. Knowledge & Invariants (The Cognitive Layer)
    // ============================================================
    knowledge {
        // --- Verified Facts (Post-Correction) ---
        fact autograd_disconnection_lesson {
            description: "Direct ct.launch calls bypass PyTorch Autograd. Weights will NOT update."
            fix: "Use Hybrid mode (Native for Train, cuTile for Eval) or explicit Autograd.Function."
            criticality: 1.0
        }

        fact bit_exact_parity_achievement {
            description: "Achieved Max Diff: 0.0000e+00 against PyTorch SDPA/LN under deterministic settings."
            confidence: 1.0
            source: "test_parity_expanded.py"
        }

        fact numerical_stability_floor_v2 {
            description: "Using -float('inf') leads to NaN in exp2(qk - m_ij). Must use -1e20."
            invariant: "StabilityFloor"
        }

        fact layernorm_contiguity_rule {
            description: "Non-pow2 outputs from cuTile kernels MUST be .contiguous() before reshape/view."
            source: "nanogpt_cutile.py correction"
        }

        // --- Transformation & Validation Rules ---
        rule "Validate Autograd Early" {
            when: "porting_new_kernel"
            apply: "Check if weight.grad is not None after first backward pass."
            priority: "Immediate"
        }

        rule "Hierarchical Parity Protocol" {
            step 1: "L1: Single Kernel functional match (Max Diff < 1e-3)"
            step 2: "L2: Single Block hidden state match"
            step 3: "L3: Full Model logits match using Weight Transplant"
        }

        rule "Indexing Consistency" {
            when: "using ct.load with shape"
            apply: "index must be in TILE units, not ELEMENT units."
        }
    }

    // ============================================================
    // 4. Operational Loops (Refined)
    // ============================================================
    agent_loop Systematic_Validator {
        step "Unit Parity" { tool.run { cmd: "python test_parity_expanded.py" } }
        step "Weight Transplant" { tool.run { cmd: "python compare_implementations.py" } }
        step "Hybrid Training" { tool.run { cmd: "python train_nanogpt_cutile.py" } }
    }

    // ============================================================
    // 5. Build Artifacts
    // ============================================================
    build {
        artifact "nanogpt_cutile_v3.py" {
            mode: "Hybrid_Autograd_Capable"
            stability: "Blackwell_Safe"
        }
        artifact "engineering_takeaways.md"
    }
}
