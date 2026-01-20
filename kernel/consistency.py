from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from .effects import ReasoningTrace

@dataclass
class StepExpectation:
    """
    Defines what we expect to happen at a specific stage of execution.
    """
    phase: str                  # Name of the phase (e.g., "Research")
    must_use_action: str        # The tool/method that must be called
    required_thought_keywords: List[str] # Keywords that must appear in reasoning

@dataclass
class PlanIR:
    """
    Intermediate Representation of the Ideal Execution Plan.
    """
    name: str
    steps: List[StepExpectation]

class ConsistencyVerifier:
    """
    Verifies that an actual Execution Trace aligns with the expected PlanIR.
    This implements the 'Round-Trip Consistency Test'.
    """
    
    def verify(self, actual_trace: List[Dict], expected_plan: PlanIR) -> Dict[str, Any]:
        print(f"⚖️ [Consistency] Verifying trace against PlanIR: '{expected_plan.name}'")
        
        # 1. Filter trace for ReasoningTrace events
        reasoning_steps = [
            t['payload'] for t in actual_trace 
            if t['name'] == 'ReasoningTrace'
        ]
        
        report = {
            "passed": True,
            "score": 0.0,
            "details": []
        }
        
        matches = 0
        
        # 2. Align Steps
        # We assume the trace should roughly follow the plan order
        trace_idx = 0
        
        for plan_step in expected_plan.steps:
            found = False
            
            # Look ahead in trace to find a match (allowing for noise/intermediate steps)
            while trace_idx < len(reasoning_steps):
                trace_item = reasoning_steps[trace_idx]
                trace_idx += 1
                
                # Check Action Match
                # trace_item.plan is a Dict, usually {'action': 'name'}
                actual_action = trace_item.plan.get('action')
                
                if actual_action == plan_step.must_use_action:
                    # Check Thought Semantic Match (Keywords)
                    thought_text = trace_item.thought.lower()
                    missing_keywords = [
                        k for k in plan_step.required_thought_keywords 
                        if k.lower() not in thought_text
                    ]
                    
                    if not missing_keywords:
                        found = True
                        matches += 1
                        report["details"].append(f"✅ Step '{plan_step.phase}': Matched action '{actual_action}' and thoughts.")
                    else:
                        # We found the action, but thoughts were insufficient.
                        # We consider this a 'Structural Match' but 'Semantic Fail'.
                        # For scoring, maybe 0.5? For now, 0 matches but we advance.
                        found = True # We found the step execution
                        report["details"].append(f"⚠️ Step '{plan_step.phase}': Action matched, but missing keywords: {missing_keywords}")
                    
                    break # Stop looking for this step, move to next plan step
            
            if not found:
                report["passed"] = False
                report["details"].append(f"❌ Step '{plan_step.phase}': Failed to find execution of '{plan_step.must_use_action}'.")
        
        # Calculate Score (Intent Recovery Rate)
        total_steps = len(expected_plan.steps)
        if total_steps > 0:
            report["score"] = matches / total_steps
        
        return report
