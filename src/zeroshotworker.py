from kernel.semantic_kernel import perform
from kernel.effects import Generate, LLMRequest

class ZeroShotWorker:
    """
    Level 0 Agent: ZeroShotWorker
    Directly maps input query to code generation without memory or planning.
    """
    
    def implement(self, query: str) -> str:
        """
        Implements the function 'implement' defined in level0.agent.spec.md
        """
        # Construct the prompt
        full_prompt = "Generate code for: " + query
        
        # Map to the standard Kernel Effect (Generate) handled by LiteLLMHandler
        # We construct a standard LLMRequest with a single user message.
        req = LLMRequest(messages=[{"role": "user", "content": full_prompt}])
        
        return perform(Generate(req))