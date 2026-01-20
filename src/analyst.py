from typing import List
from kernel.semantic_kernel import perform
from kernel.effects import Generate, LLMRequest

class Analyst:
    def extract_insights(self, documents: List[str]) -> str:
        if not documents:
            return "No documents provided to analyze."
            
        combined_text = "\n\n---\n\n".join(documents[:5]) # Limit to 5 docs
        prompt = f"""Analyze the following documents and extract key claims and evidence.
        
        DOCUMENTS:
        {combined_text}
        
        OUTPUT:
        List of insights with source references.
        """
        return perform(Generate(LLMRequest(messages=[{"role": "user", "content": prompt}]))) 

    def map_to_storyline(self, insights: str, user_intent: str, storyline_template: str) -> str:
        prompt = f"""Map the following insights to the provided storyline template based on the user's intent.
        
        INSIGHTS:
        {insights}
        
        INTENT:
        {user_intent}
        
        TEMPLATE:
        {storyline_template}
        
        OUTPUT:
        Structured outline filling the template.
        """
        return perform(Generate(LLMRequest(messages=[{"role": "user", "content": prompt}]))) 
