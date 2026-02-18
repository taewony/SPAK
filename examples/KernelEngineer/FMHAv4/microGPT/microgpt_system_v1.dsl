---
title: "MicroGPT System v1 – Tiled cuTile Transition"
source: "microgpt.py (Karpathy) + FMHAv4 Compound Knowledge"
extraction-date: 2026-02-13
tags: [MicroGPT, cuTile, TransferLearning, CompoundEngineering]
status: "active"
---

system MicroGPT_System_v1 {

    // ============================================================
    // 0. Baseline (Reference Point)
    // ============================================================
    baseline {
        implementation: "microgpt.py (Scalar Value Autograd)"
        device: "CPU (Single Core)"
        metrics {
            avg_step_time: "5.2ms" 
            target_loss_100: 2.15 
        }
    }

    // ============================================================
    // 1. Design Space (Architectural Transitions)
    // ============================================================
    design_space {
        attention_engine: ["cutile_fmha_v4", "naive_scalar"]
        normalization: ["rmsnorm", "layernorm"]
        norm_strategy: ["static_persistent", "gather_scatter"] // From rms_norm.py
        activation: ["relu", "gelu"]
        precision: ["f16", "f32"] 
        pos_encoding: ["learned_wpe", "rotary"]
    }

    // ============================================================
    // 2. Tuning Space (Inherited from FMHAv4)
    // ============================================================
    tuning_space {
        // Inherited from FMHAv4 discoveries on RTX 5070
        tile_m: [64] 
        tile_n: [64, 128]
        tile_d: [64, 128]
        
        // Pipelining parameters
        k_lat: [2, 3]
        v_lat: [4, 5]
    }

    // ============================================================
    // 3. Model & Knowledge (Compounded Intelligence)
    // ============================================================
    model {
        type GPTConfig matches { 
            n_layer: int, n_embd: int, n_head: int, block_size: int 
        }
        state current_design: "tiled_rmsnorm_relu"
    }

    knowledge {
        // --- Inherited from FMHAv4 (The Compounding Effect) ---
        fact rtx5070_fmha_optimal {
            description: "On RTX 5070, attention MUST use 64x64 tiling for peak TFLOPS."
            inherited_from: "fmha_system_v4.dsl"
            confidence: 1.0
        }

        fact tma_pipelining_preference {
            description: "V-load latency=5 is preferred for causal workloads on Blackwell."
            inherited_from: "last_engineering_trace.json"
        }

        // --- Facts from rms_norm.py ---
        fact rmsnorm_persistence_heuristic {
            description: "Use static_persistent mode when M (rows) > NUM_SMS * 2 for better wave utilization."
            source: "rms_norm.py:L240"
            confidence: 0.9
        }

        fact rmsnorm_tma_bonus {
            description: "Disabling TMA in ct.store for RMSNorm can provide up to 30% performance gain."
            source: "rms_norm.py:L195"
            confidence: 0.8
        }

        // --- Correctness Invariants ---
        invariant MathematicalEquivalence {
            assert: "Tiled ct.mma result must match scalar sum(wi*xi) within float16 epsilon."
        }

        invariant ConvergenceParity {
            assert: "Tiled loss after 100 steps must be within 5% of scalar baseline loss."
        }

        invariant ResidualCorrectness {
            assert: "x = x_residual + block(rmsnorm(x))"
            source: "microgpt.py:L106"
        }

        invariant SoftmaxStability {
            assert: "logits - max(logits) to prevent overflow"
            source: "microgpt.py:L79"
        }

        // --- Transformation Rules (Scalar -> Tiled) ---
        rule "Vectorize Linear" {
            when: "operation == 'linear' && target == 'cuTile'"
            apply: "Replace sum(wi*xi) loop with ct.mma(X, W, ACC)"
            source: "microgpt.py:L76"
        }

        rule "Fuse RMSNorm" {
            when: "normalization == 'rmsnorm'"
            apply: "Fuse scale = (ms + 1e-5)**-0.5 into the next linear load"
            source: "microgpt.py:L85"
        }
    }

    // ============================================================
    // 4. Trace Schema
    // ============================================================
    trace_schema {
        variant TraceItem {
            case Convergence { 
                step: int, 
                loss: float, 
                loss_delta_vs_baseline: float 
            }
            case Performance { 
                step_time_ms: float, 
                tflops: float, 
                speedup_vs_baseline: float 
            }
            case Correctness { 
                component: string, 
                max_diff: float, 
                passed: bool 
            }
        }
    }

    // ============================================================
    // 5. Loops (The Compound Engineering Process)
    // ============================================================

    agent_loop MicroGPT_Evolver {
        step "Lift Scalar Logic" {
            llm.query {
                prompt: "Convert the scalar 'gpt' function in microgpt.py into a tiled cuTile graph."
                output_var: "tiled_graph"
            }
        }

        step "Inject FMHA Kernel" {
            tool.write {
                path: "microgpt_kernels.py"
                content: "/* Reuse FMHAv4 kernel logic with params from tuning_space */"
            }
        }

        step "Verify Convergence" {
            tool.run { cmd: "python train_microgpt_cutile.py --steps 100" }
        }
    }

    // ============================================================
    // 6. Build
    // ============================================================
    build {
        artifact "microgpt_v1_report.md"
        artifact "microgpt_cutile_implementation.py"
    }
}
