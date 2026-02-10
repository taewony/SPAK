system_model KernelOptimizer {
  axiom: "Maximize GPU SM occupancy."
  axiom: "Reference Implementation: https://github.com/NVIDIA/cutile-python/blob/main/samples/MatMul.py"
  heuristic: "Optimize Memory Coalescing and L2 Cache reuse (Tile Swizzling)."
  prediction: "Use Tensor Cores (TF32/FP16) for >5x speedup over FP32."
}

task OptimizeMatMul {
  
  step load_existing_code: tool.read {
    path: "matmul_improved_2.py"
    output_var: existing_code
  }

  step analyze_code: llm.query {
    role: "CUDA_Engineer"
    prompt_template: """
    Analyze this CUDA Python code (cuTile). 
    Identify 3 major performance bottlenecks.
    Use the NVIDIA Reference if necessary: {{axiom.1}}
    
    Code:
    {{existing_code}}
    """
    output_var: bottlenecks
  }

  step propose_optimization: llm.query {
    role: "CUDA_Engineer"
    prompt_template: """
    Based on bottlenecks: {{bottlenecks}}
    
    Propose a specific code modification to fix the biggest bottleneck.
    Output the FULL MODIFIED FILE content.
    """
    output_var: optimized_file_content
  }

  step save_improved_code: tool.write {
    path: "matmul_improved_spak.py"
    content: "{{optimized_file_content}}"
  }

evaluation {
    check compilation: llm.query {
      role: "Judge"
      prompt_template: "Is this valid Python code? Output 1 or 0.\nCode: {{optimized_file_content}}"
      output_var: compile_score
    }
  }
}