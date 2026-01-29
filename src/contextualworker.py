from kernel.semantic_kernel import perform
from kernel.effects import Generate, LLMRequest
from typing import List

class ContextualWorker:
    """
    Level 1 Agent: ContextualWorker
    Maintains a log of interactions to provide context-aware responses.
    """
    
    def __init__(self):
        # State: Memory { logs: List[String] }
        self.logs: List[str] = []

    def implement_with_memory(self, query: str) -> str:
        """
        Implements 'implement_with_memory' from level1.agent.spec.md
        """
        # 1. Construct messages including history context
        # We format the history logs into the prompt context since we are using a simple List[String] log
        context_block = "\n".join(self.logs)
        final_prompt = f"PREVIOUS INTERACTION LOG:\n{context_block}\n\nCURRENT REQUEST:\n{query}"
        
        # 2. Perform generation using the standard Kernel Effect
        req = LLMRequest(messages=[{"role": "user", "content": final_prompt}])
        result = perform(Generate(req))
        
        # 3. Update Memory (State persistence)
        # We store the interaction pair
        interaction_log = f"User: {query} | Agent: {result}"
        self.logs.append(interaction_log)
        
        return result

    def get_memory(self) -> List[str]:
        """Helper to inspect state (for debugging/testing)"""
        return self.logs