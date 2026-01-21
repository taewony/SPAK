meta {
    name = "SPAK_Kernel"
    version = "0.6.0"
    description = "The Spec-driven Programmable Agent Kernel with Autonomous Orchestration and Round-Trip Verification"
}

// --- Operational Contract ---
contract AgentScope {
    supported_intents = [
        "Agent Synthesis: Compiling Specs into verifiable Python artifacts.",
        "Autonomous Orchestration: Driving agents via Thought-Plan-Action-Observation loops.",
        "Advanced Verification: Validating latent reasoning against symbolic PlanIR."
    ]

    success_criteria = [
        "Semantic Fidelity: Agent actions match the intended PlanIR.",
        "Effect Isolation: All side effects are mediated and logged.",
        "Traceability: 100% of LLM interactions are captured in the TraceIR."
    ]

    autonomy {
        mode = "autonomous"
        loop_interval = "2s"
        max_steps = 100
    }
}

kernel SPAK_Kernel {

    // --- 1. Algebraic Effects (The Syscall Interface) ---
    
    effect LLM {
        operation generate(prompt: String) -> String;
    }

    effect FileSystem {
        operation read(path: String) -> String;
        operation write(path: String, content: String) -> String;
        operation list_files(dir_path: String) -> List<String>;
    }
    
    effect System {
        operation recurse(spec: String, query: String) -> String;
    }

    effect ReasoningTrace {
        operation log(thought: String, plan: Map<String, Any>, raw_response: String) -> Unit;
    }

    // --- 2. Core Kernel Components ---

    component Compiler {
        description: "Parses AgentSpec DSL into Semantic IR (AST).";
        function compile_file(path: String) -> String;
    }

    component Orchestrator {
        description: "The autonomous execution engine. Implements the Thought-Action-Observation loop.";
        function run_loop(agent: Agent, goal: String) -> Result<String>;
    }

    component Verifier {
        description: "Enforces structural, behavioral, and semantic consistency.";
        function verify_structure(spec: String, src_dir: String) -> List<String>;
        function verify_behavior(test_path: String) -> List<String>;
        function verify_consistency(trace: List<String>, plan: String) -> Result<Float>;
    }

    component Builder {
        description: "Interface to LLMs for Code Synthesis, Test Generation, and Self-Repair.";

        function implement_component(spec: String, context: String) -> String;
        function generate_tests(spec: String) -> String;
        function fix_implementation(code: String, error_log: String) -> String;
    }

    component Runtime {
        description: "The isolated environment where effects are resolved and state is managed.";
        function handle_effect(effect: Effect) -> Any;
    }
    
    // --- 3. Self-Evolution Workflows ---

    workflow SelfImprovement(goal: String) {
        step Analyze {
            perform LLM.generate("Analyze kernel performance against: " + goal)
        }
        step Synthesize {
            perform LLM.generate("Propose optimized kernel implementation")
        }
        step VerifyConsistency {
            perform Verifier.verify_consistency(current_trace, "plans/kernel_upgrade.plan.yaml")
        }
    }
}