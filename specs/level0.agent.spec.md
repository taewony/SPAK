meta {
    name = "StaticResponder"
    version = "1.0"
    domain = "Conversational AI (Basic)"
    purpose = "To provide immediate, stateless responses to user inputs."
    description = "Level 0: Simple Input-Output Agent"
}

// --- Operational Contract ---
contract AgentScope {
    supported_intents = [
        "General Chat: Responding to greetings, facts, or questions."
    ]

    success_criteria = [
        "Responsiveness: Must generate a text response.",
        "Relevance: Response must address the input."
    ]
}

system StaticResponder {
    effect LLM {
        operation generate(prompt: String) -> String;
    }

    component Responder {
        description: "Responds to user queries using an LLM without maintaining state.";
        
        function reply(query: String) -> String {
            return perform LLM.generate(query)
        }
    }
}
