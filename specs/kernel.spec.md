meta {
    name = "SPAK_Kernel"
    version = "0.5.0"
    description = "The Spec-driven Programmable Agent Kernel with Reasoning Trace and Consistency Verification"
}

// --- Operational Contract ---
contract AgentScope {
    supported_intents = [
        "Agent Synthesis: Compiling Specs into executable Agents.",
        "Verification: Ensuring Agents match their Specs.",
        "Audit: Tracking and verifying agent reasoning consistency."
    ]

    success_criteria = [
        "Compilation Success: Spec parses without errors.",
        "Verification Pass: Implementation satisfies Spec and Tests.",
        "Consistency Pass: Agent execution trace matches PlanIR."
    ]

    autonomy {
        mode = "autonomous"
        loop_interval = "5s"
        max_steps = 100
    }
}

kernel SPAK_Kernel {

    // 1. Definition of Side Effects (Algebraic Effects)
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
        operation log(thought: String, plan: Map<String, String>, raw_response: String) -> Unit;
    }

    // 2. Core Components
    component Compiler {
        description: "Parses AgentSpec DSL into Semantic IR (AST)";
        function compile_file(path: String) -> String;
    }

    component Verifier {
        description: "Enforces structural, behavioral, and semantic correctness";
        function verify_structure(spec: String, src_dir: String) -> List<String>;
        function verify_behavior(test_path: String) -> List<String>;
        function verify_consistency(trace: List<String>, plan: String) -> Result<Float>;
    }

    component Builder {
        description: "Interface to Large Language Models for Code Synthesis and Repair";

        // This function uses LLM Effect
        function implement_component(spec: String, context: String) -> String {
            perform LLM.generate(context + spec)
        }

        function generate_tests(spec: String) -> String {
            perform LLM.generate("Create tests for " + spec)
        }

        function fix_implementation(code: String, error_log: String) -> String {
            perform LLM.generate("Fix this code: " + code + " Error: " + error_log)
        }
    }

    component Runtime {
        description: "Execution environment for built agents";
        function run_component(name: String) -> String;
    }
    
    // 3. Self-Evolution Workflow
    workflow SelfImprovement(goal: String) {
        step Analyze {
            perform LLM.generate("Analyze current kernel performance against: " + goal)
        }
        
        step Synthesize {
            perform LLM.generate("Propose new kernel code")
        }
        
        step Verify {
            // New verification step in the loop
            perform Verifier.verify_structure("specs/kernel.spec.md", "kernel")
        }
    }
}
