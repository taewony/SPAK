system Attention_System_v0 {

    // ============================================================
    // 1. Design Space (Initial/Generic)
    // ============================================================
    design_space {
        softmax_scheme: ["naive"]
        mask_type: ["causal", "none"]
        precision: ["f16", "f32"]
    }

    // ============================================================
    // 2. Tuning Space (Standard Search Space)
    // ============================================================
    tuning_space {
        block_size: [32, 64, 128]
        num_warps: [4, 8]
    }

    // ============================================================
    // 3. Model & Knowledge (Semantic Goal: "What")
    // ============================================================
    model {
        type Task matches { head_dim: int, seq_len: int }
        state objective: "Functional Correctness & Basic Performance"
    }

    knowledge {
        // High-level "What" invariants
        invariant Correctness {
            assert: "The output of FMHA must match the reference PyTorch implementation."
        }

        invariant Scalability {
            assert: "Memory usage must be O(N) rather than O(N^2)."
        }

        // Generic rule
        rule "Basic Optimization" {
            when: "is_slow"
            apply: "Apply standard tiling and loop unrolling."
        }

        fact is_slow(t: TraceItem.Performance) {
            return t.tflops < 10.0
        }
    }

    // ============================================================
    // 4. Trace Schema
    // ============================================================
    trace_schema {
        variant TraceItem {
            case Performance { tflops: float }
            case Correctness { passed: boolean }
        }
    }

    // ============================================================
    // 5. Loops
    // ============================================================
    agent_loop Attention_Architect {
        step "Setup Baseline" {
            tool.write { path: "fmha_v0.py", content: "# Initial implementation" }
            tool.run { cmd: "python fmha_v0.py" }
        }
    }

    engineering_loop Attention_Tuner {
        parameter BlockSize: [32, 64]
        measure {
            cmd: "python fmha_v0.py --block {{BlockSize}}"
            metric: "tflops"
            objective: "maximize"
        }
    }
}
