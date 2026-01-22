from typing import List, Dict, Any
from dataclasses import dataclass
import json
from kernel.semantic_kernel import perform
from kernel.effects import Generate, LLMRequest

@dataclass
class Insight:
    claim: str
    evidence: str
    source: str
    confidence: float

class Analyst:
    def extract_insights(self, documents: List[str]) -> List[Insight]:
        if not documents:
            return []
            
        # Context management: Strict limit to prevent CUDA OOM on smaller GPUs
        # TODO: Implement Map-Reduce for large sets
        combined_text = "\n\n---\n\n".join(documents[:1]) 
        
        prompt = f"""You are a Research Analyst. Extract key insights from the documents.
        
        DOCUMENTS:
        {combined_text}
        
        REQUIREMENTS:
        1. Extract claims, evidence, source reference, and confidence score (0.0-1.0).
        2. Output strictly in JSON format as a list of objects:
        [
          {{
            "claim": "...",
            "evidence": "...",
            "source": "...",
            "confidence": 0.9
          }}
        ]
        """
        
        response = perform(Generate(LLMRequest(
            messages=[{"role": "user", "content": prompt}]
        )))
        
        try:
            # Clean JSON
            clean_json = response.strip()
            if "```json" in clean_json:
                clean_json = clean_json.split("```json")[1].split("```")[0]
            elif "```" in clean_json:
                clean_json = clean_json.split("```")[1].split("```")[0]
            
            clean_json = clean_json.strip()
            # Fix common LLM JSON syntax errors (e.g. escaped underscores)
            clean_json = clean_json.replace("\\_", "_")
            
            # Handle potential trailing characters or intro text if LLM is chatty
            if not clean_json.startswith("["):
                # Try to find the list
                start = clean_json.find("[")
                end = clean_json.rfind("]")
                if start != -1 and end != -1:
                    clean_json = clean_json[start:end+1]

            data = json.loads(clean_json)
            insights = [Insight(**item) for item in data]
            return insights
        except Exception as e:
            print(f"Error parsing insights: {e}")
            print(f"Raw Response: {response}")
            return []

    def create_structured_outline(self, insights: List[Insight], intent: str, structure_type: str) -> str:
        # Convert structured insights back to text for the prompt
        insights_text = json.dumps([i.__dict__ for i in insights], indent=2)
        
        prompt = f"""You are a Structural Architect. Create a detailed outline for the document.
        
        GOAL (INTENT):
        {intent}
        
        REQUIRED STRUCTURE:
        {structure_type}
        
        Available Insights (Ingredients):
        {insights_text}
        
        INSTRUCTIONS:
        1. Map the available insights into the sections defined by the Required Structure.
        2. Ensure a logical flow of arguments.
        3. Output a clean Markdown outline (headings and bullet points).
        """
        return perform(Generate(LLMRequest(messages=[{"role": "user", "content": prompt}])))
