from typing import List, Dict, Any
from dataclasses import dataclass
import json
from kernel.semantic_kernel import perform
from kernel.effects import Generate, LLMRequest

@dataclass
class EvaluationResult:
    status: str      # "PASS" or "FAIL"
    feedback: str    # Actionable feedback
    score: float     # 0.0 to 10.0

class Reviewer:
    """
    The 'Taster' component.
    Evaluates artifacts against success criteria using LLM as a judge.
    """

    def evaluate(self, draft: str, criteria: List[str]) -> EvaluationResult:
        if not draft:
            return EvaluationResult(status="FAIL", feedback="Draft is empty.", score=0.0)

        criteria_str = "\n".join([f"- {c}" for c in criteria])

        prompt = f"""You are a Strict Reviewer. Evaluate the provided draft against the criteria.

DRAFT:
{draft}

CRITERIA:
{criteria_str}

REQUIREMENTS:
1. Determine if the draft meets the criteria (PASS/FAIL).
2. Provide concise, actionable feedback for improvement if FAIL, or praise if PASS.
3. Assign a score from 0.0 to 10.0.
4. Output strictly in JSON format:
{{
  "status": "PASS" or "FAIL",
  "feedback": "...",
  "score": 8.5
}}
"""
        response = perform(Generate(LLMRequest(
            messages=[{"role": "user", "content": prompt}]
        )))

        try:
            # Robust JSON extraction
            clean_json = response.strip()
            if "```json" in clean_json:
                clean_json = clean_json.split("```json")[1].split("```")[0]
            elif "```" in clean_json:
                clean_json = clean_json.split("```")[1].split("```")[0]
            
            clean_json = clean_json.strip()
            
            # Handle potential surrounding text
            if not clean_json.startswith("{"):
                start = clean_json.find("{")
                end = clean_json.rfind("}")
                if start != -1 and end != -1:
                    clean_json = clean_json[start:end+1]

            data = json.loads(clean_json)
            
            return EvaluationResult(
                status=data.get("status", "FAIL"),
                feedback=data.get("feedback", "Error parsing feedback."),
                score=float(data.get("score", 0.0))
            )

        except Exception as e:
            print(f"Error parsing evaluation: {e}")
            print(f"Raw Response: {response}")
            return EvaluationResult(
                status="FAIL", 
                feedback=f"System Error: Could not parse reviewer response. {e}", 
                score=0.0
            )
