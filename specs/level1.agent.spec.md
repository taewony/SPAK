meta {
    name = "ContextualCoder"
    version = "1.0"
    domain = "Iterative Code Generation"
    purpose = "Level 1: Recalls previous errors but does not perform formal planning."
    description = "The Contextual Coder uses a memory buffer to avoid repeating past mistakes, but still maps input to output directly."
}

// --- System Model ---
system_model ContextualPrinciples {
    supported_intents = [
        "Iterative Refinement: Improving code based on previous execution logs."
    ]

    heuristic: "Reference similar past solutions when applicable"
    prediction: "With more history, accuracy improves"
}

system ContextualCoder {
    effect LLM {
        operation generate_with_context(prompt: String, history: List<String>) -> String;
    }

	// --- Core Component ---
    component ContextualWorker {
        description: "Generates code while maintaining a simple memory of previous interactions.";
        
        state Memory {
            logs: List[String]
        }

        function implement_with_memory(query: String) -> String {
            return perform LLM.generate_with_context(query, state.logs)
        }
    }
}
