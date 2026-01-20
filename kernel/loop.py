import json
from typing import Any, List, Dict
from .semantic_kernel import perform
from .effects import Generate, LLMRequest, ReasoningTrace, TraceLog

class AutonomousLoop:
    """
    Executes an autonomous Thought-Action-Observation loop for an agent.
    Strictly adheres to the 'LLM as Effect' principle.
    """
    def __init__(self, agent_instance: Any, model_name: str = "qwen2.5:3b"):
        self.agent = agent_instance
        self.model_name = model_name
        self.max_steps = 10

    def run(self, goal: str):
        print(f"🚀 [Loop] Starting autonomous execution for goal: {goal}")
        
        history = []
        for step in range(self.max_steps):
            print(f"\n--- Step {step + 1} ---")
            
            # 1. Context Gathering
            context = self._get_agent_context()
            available_methods = self._get_available_methods()
            
            # 2. Planning (LLM Call via Effect)
            prompt = f"""
            Goal: {goal}
            Current State: {context}
            History: {history}
            Available Actions: {available_methods}
            
            You are the decision engine. Analyze the situation and choose the next action.
            
            Output strictly strictly valid JSON:
            {{
              "thought": "Analysis of the situation...",
              "plan": "High-level plan description...",
              "action": "method_name",
              "arguments": {{ "arg_name": "value" }},
              "is_final": false,
              "final_result": null
            }}
            """
            
            try:
                # LLM is invoked as a pure function via the Kernel
                llm_response = perform(Generate(LLMRequest(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.model_name
                )))
                
                # Parse Decision
                # Simple cleanup for markdown code blocks if present
                clean_response = llm_response.replace("```json", "").replace("```", "").strip()
                decision = json.loads(clean_response)
                
                # 3. Trace Logging (Formal Artifact)
                perform(ReasoningTrace(TraceLog(
                    thought=decision.get('thought', 'No thought provided'),
                    plan={"action": decision.get('action')},
                    raw_response=llm_response
                )))
                
                # 4. Termination Check
                if decision.get('is_final'):
                    print(f"🏁 Goal achieved: {decision.get('final_result')}")
                    return decision.get('final_result')

                # 5. Execution
                action_name = decision['action']
                args = decision.get('arguments', {})
                
                if not hasattr(self.agent, action_name):
                    raise ValueError(f"Unknown action: {action_name}")
                
                print(f"🎬 Executing: {action_name}({args})")
                method = getattr(self.agent, action_name)
                
                # The method itself calls 'perform(...)' internally
                result = method(**args)
                
                print(f"👁️ Observation: {str(result)[:150]}...")
                
                history.append({
                    "step": step,
                    "thought": decision.get('thought'),
                    "action": action_name,
                    "result": str(result)
                })
                
            except json.JSONDecodeError:
                print(f"❌ LLM produced invalid JSON: {llm_response[:100]}...")
            except Exception as e:
                print(f"❌ Loop Error: {e}")
                break
        
        print("⚠️ Reached max steps without final result.")

    def _get_agent_context(self) -> str:
        # Check if agent has a 'state' attribute or component states
        if hasattr(self.agent, 'state'):
            return str(self.agent.state)
        # Fallback: check for 'context' or similar
        return "No explicit state."

    def _get_available_methods(self) -> List[str]:
        # Filter out dunder methods and private methods
        return [m for m in dir(self.agent) if not m.startswith('_') and callable(getattr(self.agent, m))]