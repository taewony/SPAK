import json
from typing import List, Dict, Tuple
from difflib import SequenceMatcher
from .mas import MinimalAgentSpec, MASTask, MASInvariant

class MetricEngine:
    """
    Calculates consistency, grounding, and fidelity metrics
    by comparing Original Specs vs. Recovered Specs.
    """

    def calculate_consistency_index(self, original: MinimalAgentSpec, recovered: MinimalAgentSpec) -> Dict[str, float]:
        """
        Compares two MAS objects and returns similarity scores (0.0 - 1.0).
        """
        
        # 1. Intent Semantic Similarity (Mocked with SequenceMatcher for now)
        intent_score = SequenceMatcher(None, original.intent.description, recovered.intent.description).ratio()

        # 2. Task Alignment
        # Check how many original tasks are present in recovered (recall)
        orig_tasks = {t.name for t in original.tasks}
        rec_tasks = {t.name for t in recovered.tasks}
        
        intersection = orig_tasks.intersection(rec_tasks)
        task_recall = len(intersection) / len(orig_tasks) if orig_tasks else 1.0
        task_precision = len(intersection) / len(rec_tasks) if rec_tasks else 1.0
        task_f1 = 2 * (task_precision * task_recall) / (task_precision + task_recall) if (task_precision + task_recall) > 0 else 0.0

        # 3. Invariant Preservation
        # This is harder to match exactly, so we look for keyword overlap or embedding similarity
        # Here we use a naive set overlap of descriptions
        orig_invs = {inv.description for inv in original.invariants}
        rec_invs = {inv.description for inv in recovered.invariants}
        
        # A simple overlap ratio for now
        # In a real system, we would use vector embeddings to check semantic coverage
        inv_score = SequenceMatcher(None, " ".join(orig_invs), " ".join(rec_invs)).ratio()

        return {
            "consistency_index": (intent_score + task_f1 + inv_score) / 3,
            "intent_similarity": intent_score,
            "task_alignment_f1": task_f1,
            "invariant_preservation": inv_score
        }

    def calculate_grounding_index(self, trace_log: List[Dict], artifacts: List[str]) -> float:
        """
        Measures how much of the final artifact is traceable to the source items in the log.
        Baseline: 1.0 means every artifact claim has a corresponding 'ExtractEvidence' step.
        """
        # This requires analyzing the trace for 'Item(Evidence)' creation
        # and checking if the 'Synthesize' step references those Items.
        
        evidence_count = sum(1 for step in trace_log if step.get("task") == "ExtractEvidence")
        if evidence_count == 0:
            return 0.0
            
        # Simplified Logic: if we have evidence, do we have a synthesis step?
        synthesis_count = sum(1 for step in trace_log if step.get("task") == "SynthesizeFinalArtifact")
        
        if synthesis_count > 0:
            return 1.0 # Ideal case for this mock
        return 0.0

class SpecRecoverer:
    """
    Simulates the 'External Foreign LLM' recovering a Spec from a Trace.
    """
    def recover_from_trace(self, trace_log_path: str) -> MinimalAgentSpec:
        # In a real implementation, this would:
        # 1. Read the trace JSONL
        # 2. Feed it to an LLM with the prompt: "Reverse engineer the agent spec from this log..."
        # 3. Parse the output into MinimalAgentSpec
        
        # For now, return a placeholder or mock
        return None 
