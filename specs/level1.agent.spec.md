meta {
    name = "ContextBot"
    version = "1.0"
    domain = "Conversational AI (Contextual)"
    purpose = "To engage in multi-turn conversations by maintaining dialogue history."
    description = "Level 1: Context-Aware Agent"
}

// --- Operational Contract ---
contract AgentScope {
    supported_intents = [
        "Contextual Chat: Answering questions based on previous turns.",
        "Conversation Tracking: Remembering user details mentioned earlier."
    ]

    success_criteria = [
        "Continuity: Responses must reflect prior context.",
        "Coherence: The conversation flow must be logical."
    ]
}

system ContextBot {
    effect LLM {
        operation generate(prompt: String) -> String;
    }

    component Assistant {
        description: "Maintains conversation history and responds with context.";

        state Conversation {
            history: List<String>
        }

        function chat(message: String) -> String {
            # In a real implementation, this would append to history
            # and send the full context to the LLM.
            prompt = "History: " + str(state.history) + "\nUser: " + message
            response = perform LLM.generate(prompt)
            return response
        }
    }
}

