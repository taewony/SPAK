meta {
    name = "ZeroShotCoder"
    version = "1.0"
    domain = "Direct Code Generation"
    purpose = "Level 0: Maps Input directly to Code without simulation or planning."
    description = "The Zero-Shot Coder performs no latent simulation. It assumes a static mapping from request to implementation."
}

// --- System Model ---
system_model ZeroShotPrinciples {
	
    supported_intents = [
        "Direct Implementation: Generating snippets based on prompts."
    ]
}

system ZeroShotCoder {
    // Level 0 uses a minimal LLM effect without planning wrappers
    effect LLM {
        operation generate(prompt: String) -> String;
    }

	// --- Core Component ---
    component ZeroShotWorker {
        description: "Directly generates code from a prompt without internal verification.";
        
        function implement(query: String) -> String {
            // Note: No 'synthesize_plan' step here.
            return perform LLM.generate("Generate code for: " + query)
        }
    }
}