meta {
  name        = "ChatBotSystem"
  version     = "1.0"
  description = "A conversational agent that uses LLM effects"
}

system ChatBotSystem {

    // Define the capability to call LLM
    effect LLM {
        operation generate(prompt: String) -> String;
    }

    component ChatBot {
        description: "A simple conversational agent with memory";
        
        state ChatState {
            history: List[String]
            persona: String
        }

        # Initialize with a persona
        function set_persona(persona: String) -> String {
            return "Persona set to " + persona
        }

        # The core loop: Add user msg -> Call LLM -> Add response -> Return
        function chat(user_message: String) -> String {
            perform LLM.generate(user_message)
        }
        
        function get_history() -> List[String];
    }
}