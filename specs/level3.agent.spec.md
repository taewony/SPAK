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
        operation think(goal: String) -> String;
        operation decide(plan: String, context: String) -> String;
        operation reflect(reason: String) -> String;
        operation revise(plan: String) -> String;
    }
    
    effect User {
        operation listen() -> String;
        operation reply(message: String) -> String;
    }

    component Coach {
        description: "A coaching agent that plans, executes, and self-corrects.";
        
        state Session {
            goal: String
            plan: String
            history: List<String>
            user_feeling: String
        }

        function configure_session(goal: String) -> String;
        
        workflow StartSession(feeling: String) {
            step Greet {
                perform User.reply("Hello! I am your Coach. How are you feeling today?")
            }
            step Listen {
                # Only performed if feeling is not provided
                perform User.listen()
            }
            step Plan {
                perform LLM.think(state.goal)
            }
        }
    }
}