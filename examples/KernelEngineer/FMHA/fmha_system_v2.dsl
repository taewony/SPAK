system FMHA_System_v2 {

    // 1. Model & Knowledge (Shared)
    model {
        type Config matches { tile_size: int, stages: int }
        state current_best: Config
    }

    trace_schema {
        variant TraceItem {
            case Performance { tflops: float, latency: float }
            case Error { msg: string }
        }
    }

    knowledge {
        fact is_converged(t: TraceItem.Performance) {
            return t.tflops > 100.0
        }
    }

    // 2. Agent Loop: The "Architect"
    // Responsible for Logic & Code Generation
    agent_loop Kernel_Architect {
        
        step "Generate Invariant Test" {
            // Write the Python Simulation to prove math
            tool.write { path: "fmha_step1_sim.py" }
            tool.run { cmd: "python fmha_step1_sim.py" }
        }

        step "Generate CUDA Kernel" {
            // Write the Initial Fused Kernel
            llm.query { 
                prompt: "Write a CuTile kernel for FMHA..." 
                output_var: "kernel_code"
            }
            tool.write { path: "fmha_kernel.py", content: "{{kernel_code}}" }
        }

        step "Review Optimization Results" {
            // Read traces from Engineering Loop
            // If satisfied, finalize. If not, refine kernel.
            if "is_converged(trace.latest)" {
                emit "Optimization Complete"
            }
        }
    }

    // 3. Engineering Loop: The "Tuner"
    // Responsible for Parameter Sweep & Measurement (No LLM)
    engineering_loop Kernel_Tuner {
        
        // Define the Search Space provided by the Architect
        parameter Tile_M: [64, 128]
        parameter Tile_N: [32, 64, 128]
        parameter PipelineStages: [2, 3]

        // Execution Protocol
        measure {
            cmd: "python fmha_kernel.py --tile_m {{Tile_M}} --tile_n {{Tile_N}} --stages {{PipelineStages}}"
            metric: "tflops"
            objective: "maximize"
        }
        
        // Output: Generates a stream of TraceItem.Performance
    }

    // 4. Service Loop: The "API" (Hypothetical)
    // If this kernel were served in production
    service_loop Inference_Service {
        on request(query_tensor) {
            tool.run { cmd: "./run_inference {{query_tensor}}" }
            emit "response"
        }
    }

    build {
        artifact "fmha_optimized.cu" {
            source: "fmha_kernel.py"
            config: "engineering_loop.best_params"
        }
    }
}