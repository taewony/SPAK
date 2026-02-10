system MatMul_Optimizer {

    // ============================================================
    // 1. Model
    // ============================================================
    model {
        type Matrix matches { shape: [M, N, K], dtype: "fp16", layout: "RowMajor" }
        state best_tflops: float = 0.0
    }

    // ============================================================
    // 2. Trace Schema
    // ============================================================
    trace_schema {
        variant TraceItem {
            case Performance {
                step_name: string
                tflops: float
                speedup: float
            }
            case Analysis {
                bottleneck: "DRAM_BW" | "L2_Conflict" | "Pipeline_Stall"
            }
        }
    }

    // ============================================================
    // 3. Knowledge (MatMul Specific)
    // ============================================================
    knowledge {
        // Facts
        fact is_l2_bound(t: TraceItem.Performance) {
            // Hypothesis: Swizzling helps if we are L2 bound
            return t.speedup < 0.85
        }

        // Rules
        rule "Block Swizzling" {
            when: "is_l2_bound"
            apply: "Remap block coordinates to maximize L2 tile reuse."
        }

        rule "Double Buffering" {
            when: "Pipeline_Stall"
            apply: "Overlap Global Memory Load with Math using asynchronous copy."
        }

        hint "AutoTuning" {
            trigger: "Architectural Unknowns"
            suggestion: "Sweep Tile Sizes [64, 128, 256] and Occupancy [1, 2, 4]."
        }
    }

    // ============================================================
    // 4. Loops
    // ============================================================
    agent_loop Kernel_Engineer {
        
        step "Level 1: Naive Tiling" {
            tool.run { cmd: "python step1_naive_tiling.py" }
            emit TraceItem.Performance
        }

        step "Level 2: Occupancy Optimization" {
            tool.run { cmd: "python step2_occupancy.py" }
            emit TraceItem.Performance
        }

        step "Level 3: Swizzling" {
            // Apply L2 Optimization
            tool.run { cmd: "python step3_swizzling.py" }
            emit TraceItem.Performance
        }

        step "Level 4: Pipelining" {
            // Apply Latency Hiding
            tool.run { cmd: "python step4_pipelining.py" }
            emit TraceItem.Performance
        }

        step "Level 5: Auto-Tuning" {
            // Find hardware sweetspot
            tool.run { cmd: "python step5_autotuner.py" }
            emit TraceItem.Performance
        }
    }

    // ============================================================
    // 5. Build
    // ============================================================
    build {
        artifact "Final_MatMul_Report.md" {
            generator: "python generate_final_report.py"
            input: trace.all
        }
    }
}
