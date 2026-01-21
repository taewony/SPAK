meta {
    name = "CoachingAgent"
    version = "1.0"
    domain = "Personal Development & Planning"
    purpose = "To guide users through goals using planning, execution, and reflection loops."
    description = "Level 3: Planning Agent with Workflow"
}

// --- Operational Contract ---
contract AgentScope {
    supported_intents = [
        "Goal Setting: Defining objectives.",
        "Coaching Session: Interactive guidance and feedback.",
        "Self-Correction: Refining plans based on reflection."
    ]

    success_criteria = [
        "Adaptability: Plan evolves based on user input.",
        "Goal Alignment: Actions must move towards the session goal."
    ]

    autonomy {
        mode = "autonomous"
        loop_interval = "5s"
        max_steps = 50
    }
}

system CoachingAgent {
    effect LLM {
        // 'think' captures the latent reasoning process (THOUGHT)
        operation think(context: String) -> String;
        
        // 'plan' translates thought into an executable or stateful strategy (PLAN)
        operation plan(thought: String) -> String;
        
        // 'revise' updates the current plan based on new reasoning
        operation revise(current_plan: String, reason: String) -> String;
        
        operation respond(plan: String, history: List<String>) -> String;
    }
    
    effect User {
        operation listen() -> String;
        operation reply(message: String) -> String;
    }

    component Coach {
        description: "A planning agent that maintains an explicit strategy. It thinks before acting and revises its plan based on user feedback.";
        
        state Session {
            goal: String
            plan: String // Explicit strategy state
            history: List<String>
        }
        
        function start(feeling: String) -> String;
        function end() -> String;
    }
}