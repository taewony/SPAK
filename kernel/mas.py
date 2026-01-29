from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

@dataclass
class Intent:
    description: str
    domain: str

@dataclass
class MASTask:
    name: str
    description: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)

@dataclass
class MASItem:
    name: str
    schema: Dict[str, str] # field_name -> description

@dataclass
class MASInvariant:
    description: str
    scope: str = "global" # or task-specific

@dataclass
class SuccessCriteria:
    metric: str
    threshold: str

@dataclass
class MinimalAgentSpec:
    """
    The Canonical Form for Agent Specifications.
    Used for comparison, recovery, and alignment metrics.
    """
    name: str
    intent: Intent
    tasks: List[MASTask] = field(default_factory=list)
    items: List[MASItem] = field(default_factory=list)
    invariants: List[MASInvariant] = field(default_factory=list)
    criteria: List[SuccessCriteria] = field(default_factory=list)

    def to_prompt_text(self) -> str:
        """Render as a standardized text for LLM Consumption/Generation"""
        lines = [f"AGENT_SPEC_NORMAL_FORM {self.name} {{"]
        
        lines.append(f"  INTENT {{")
        lines.append(f"    {self.intent.description}")
        lines.append(f"  }}")

        lines.append(f"  TASKS {{")
        for t in self.tasks:
            lines.append(f"    TASK {t.name}: {t.description}")
        lines.append(f"  }}")

        lines.append(f"  ITEMS {{")
        for i in self.items:
            lines.append(f"    ITEM {i.name} {i.schema}")
        lines.append(f"  }}")

        lines.append(f"  INVARIANTS {{")
        for inv in self.invariants:
            lines.append(f"    {inv.description}")
        lines.append(f"  }}")

        lines.append(f"  SUCCESS_CRITERIA {{")
        for c in self.criteria:
            lines.append(f"    {c.metric}: {c.threshold}")
        lines.append(f"  }}")

        lines.append("}")
        return "\n".join(lines)
