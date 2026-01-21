from typing import List, Dict
from kernel.semantic_kernel import perform
from kernel.effects import Generate, LLMRequest, ReasoningTrace, TraceLog

class PersonaChatBot:
    """A minimal persona-driven conversational agent"""

    def __init__(self, persona: str = "financial consultant"):
        self.state = {
            "persona": persona,
            "history": [],
            "last_user_input": "",
            "last_response": ""
        }

    def set_persona(self, new_persona: str) -> str:
        self.state["persona"] = new_persona
        self.state["history"] = [] # Reset history on persona change as per spec
        return "Persona updated and conversation history reset."

    def chat(self, user_input: str) -> str:
        self.state["last_user_input"] = user_input
        
        # --- Phase 1: Think (Interpret intent under persona constraints) ---
        perform(ReasoningTrace(TraceLog(
            thought=f"User said: '{user_input}'. I need to respond as '{self.state['persona']}'.",
            plan={"action": "chat", "persona": self.state["persona"]}
        )))

        # --- Phase 2: Respond (Generate reply) ---
        history_text = "\n".join(self.state["history"][-5:]) # Keep last 5 turns context
        
        # Construct messages with proper System Role separation
        messages = [
            {"role": "system", "content": f"You are a {self.state['persona']}. Respond strictly in character."},
            {"role": "user", "content": f"History:\n{history_text}\n\nUser: {user_input}"}
        ]
        
        response = perform(Generate(LLMRequest(messages=messages)))
        
        # --- Phase 3: Reflect (Update Memory) ---
        self.state["last_response"] = response
        self.state["history"].append(f"User: {user_input}")
        self.state["history"].append(f"Agent: {response}")
        
        return response

    def get_history(self) -> List[str]:
        return self.state["history"]

    def start(self) -> str:
        """Interactive helper for the kernel 'run' command."""
        print(f"[System]: Starting ChatBot with persona: '{self.state['persona']}'")
        print("[System]: Type 'exit' to quit.")
        print("[System]: Type '/persona [role]' to change the agent's personality.")
        
        while True:
            user_input = input("\n[User]: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            
            if user_input.startswith("/persona"):
                new_persona = user_input.replace("/persona", "").strip()
                if new_persona:
                    msg = self.set_persona(new_persona)
                    print(f"[System]: {msg}")
                    continue
            
            response = self.chat(user_input)
            print(f"\n[Agent]: {response}")
            
        return "Chat session ended."