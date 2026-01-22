from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from .compiler import SystemSpec, ComponentSpec

@dataclass
class InvariantResult:
    name: str
    status: str  # PASS, FAIL, UNVERIFIED
    details: str

class DomainInvariantChecker:
    """
    Validates the alignment between symbolic execution and engineering laws (Domain Correctness).
    Matches the 'DomainInvariantChecker' component in dual_validation.spec.md.
    """

    def check_invariants(self, trace: List[Dict], spec: SystemSpec) -> List[InvariantResult]:
        results = []
        
        # 1. Identify final state or relevant artifacts from trace
        # For now, we look at the last "Result" in the trace as the 'final state' proxy
        final_result = None
        for entry in reversed(trace):
            if entry.get('result') is not None:
                final_result = entry['result']
                break
        
        state_summary = f"Final Result: {str(final_result)[:100]}..." if final_result else "No Result"

        # 2. Iterate over all components and their invariants
        for comp in spec.components:
            for inv_str in comp.invariants:
                # In a full implementation, we would parse 'inv_str' into executable logic.
                # For this version, we treat NL invariants as "Manual Audit Required" or 
                # use simple heuristic checks if they follow a specific pattern (e.g. "Result must contain X")
                
                status = "UNVERIFIED"
                details = f"Requires manual or heuristic check: '{inv_str}'"

                # Simple Heuristic: "Must contain X"
                if "must contain" in inv_str.lower() and final_result:
                    required_term = inv_str.split("contain")[-1].strip().strip('"').strip("'")
                    
                    # Handle parenthesis list: "all required sections (A, B, C)"
                    import re
                    list_match = re.search(r'\((.*?)\)', required_term)
                    
                    if list_match:
                        # Check all items in the list
                        required_items = [i.strip() for i in list_match.group(1).split(',')]
                        missing = [i for i in required_items if i.lower() not in str(final_result).lower()]
                        
                        if not missing:
                            status = "PASS"
                            details = f"Found all required items: {required_items}"
                        else:
                            status = "FAIL"
                            details = f"Missing items: {missing}"
                    
                    # Fallback to simple string match
                    elif required_term.lower() in str(final_result).lower():
                        status = "PASS"
                        details = f"Found '{required_term}' in result."
                    else:
                        status = "FAIL"
                        details = f"Missing '{required_term}' in result."

                # Simple Heuristic: "Output must be"
                elif "output must be" in inv_str.lower() and final_result:
                    required_val = inv_str.split("be")[-1].strip().strip('"').strip("'")
                    # exact match check
                    if str(final_result).strip() == required_val:
                         status = "PASS"
                         details = "Exact match."
                    else:
                         status = "FAIL"
                         details = f"Expected '{required_val}', got '{final_result}'"

                results.append(InvariantResult(
                    name=f"{comp.name}::{inv_str[:30]}...",
                    status=status,
                    details=details
                ))

        return results
