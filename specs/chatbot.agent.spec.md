meta {
  name        = "PersonaChatBot"
  version     = "1.0"
  description = "A minimal persona-driven conversational agent"
}

system ChatBotSystem {

    effect LLM {
        operation think(prompt: String) -> String;
        operation reply(thought: String) -> String;
    }

    component PersonaChatBot {
        description: "A persona-based conversational responder";
        
        state ChatState {
            persona: String
            history: List<String>
            last_user_input: String
            last_response: String
        }

        function set_persona(new_persona: String) -> String;

        function chat(user_input: String) -> String;
        
        function get_history() -> List[String];
    }
}