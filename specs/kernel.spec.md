meta {
    name = "SPAK_Kernel"
    version = "0.7.0"
    description = "A lightweight runtime for Spec-Guided Latent Planning and Re-entrant Execution."
}

contract KernelScope {
    supported_intents = [
        "Spec Compilation: Lark-based parsing of AgentSpec into System Prompts.",
        "Latent Planning: Injection of 'System Models' into LLM context.",
        "Soft-Sandboxing: Restricted Python exec() for effect handling."
    ]
}

kernel SPAK_Kernel {

    // --- 1. The "Cognitive" Effects (Interfaces to the LLM) ---
    
    effect Planner {
        // Injects the 'system_model' and requests a structured Plan-IR
        operation synthesize_plan(goal: String, model_context: String) -> PlanIR;
        
        // Asks the LLM to critique its own result against the expected outcome
        operation reflect(result: String, expectation: String) -> Reflection;
    }

    effect ToolRuntime {
        // Soft-sandbox: executes Python code strings within a restricted namespace
        operation exec_restricted(code: String, allowed_modules: List<String>) -> Result;
        
        // Standard File I/O (mediated)
        operation read_file(path: String) -> String;
        operation write_file(path: String, content: String) -> String;
    }

    // --- 2. Core Components ---

    component SpecLoader {
        description: "Lark-based Parser. Converts .spec files into Runtime Objects.";
        function extract_system_model(spec_path: String) -> String; 
        function extract_invariants(spec_path: String) -> List<Constraint>;
    }

    component Orchestrator {
        description: "The Level 4 Loop Engine.";
        
        // The main loop for "Re-planning if result is not good enough"
        function run_adaptive_loop(agent_name: String, goal: String) -> Artifact;
    }

    component SafetyMonitor {
        description: "The 'Soft' Guardrail.";
        // Checks generated Python code for banned AST nodes (e.g., 'import os', 'subprocess')
        function scan_code_safety(code: String) -> Bool;
    }
}
