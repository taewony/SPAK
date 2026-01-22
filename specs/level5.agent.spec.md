meta {
    name = "MetaSolver"
    version = "1.0"
    domain = "Recursive Problem Solving"
    purpose = "To solve arbitrarily complex problems by dynamically spawning sub-agents."
    description = "Level 5: Recursive Agent (Fractal Solver)"
}

// --- Operational Contract ---
contract AgentScope {
    supported_intents = [
        "Complex Reasoning: Breaking down hard problems.",
        "Dynamic Delegation: Spawning sub-agents as needed."
    ]

    success_criteria = [
        "Recursion: Successfully calling sub-agents.",
        "Synthesis: Aggregating sub-results into a final answer."
    ]

    autonomy {
        mode = "autonomous"
        loop_interval = "2s"
        max_steps = 200
    }
}

system MetaSolver {
    effect System {
        # Spawns a new agent instance with fresh context
        operation recurse(spec_path: String, query: String) -> String;
    }

    component Orchestrator {
        description: "Solves complex problems by delegating to specialized sub-agents.";

        workflow SolveBigProblem(problem: String) {
            step Decompose {
                # In reality, LLM would decide which spec to use.
                # Here we hardcode delegating a math sub-problem to the CalculatorAgent.
                
                # "I need to calculate 25 * 4 as part of the big problem"
                # Corrected path to Level 2 Spec
                sub_result = perform System.recurse("specs/level2.agent.spec.md", "multiply 25 by 4")
            }
        }
    }
}