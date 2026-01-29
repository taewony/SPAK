from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

@dataclass
class Step:
    id: str
    type: str # 'tool', 'llm', 'metric_tool', 'metric_llm'

@dataclass
class ToolStep(Step):
    tool_name: str
    params: Dict[str, Any] = field(default_factory=dict)
    output_var: Optional[str] = None

@dataclass
class LLMStep(Step):
    role: str
    prompt_template: str
    output_var: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Metric:
    id: str
    logic: Step # Holds the ToolStep or LLMStep

@dataclass
class SuccessCondition:
    check_type: str 
    args: List[Any] = field(default_factory=list)

@dataclass
class Task:
    name: str
    steps: List[Step] = field(default_factory=list)
    evaluation: List[Metric] = field(default_factory=list)
    success_criteria: List[SuccessCondition] = field(default_factory=list)
    system_model: Optional['SystemModel'] = None

@dataclass
class SystemModel:
    name: str
    axioms: List[str] = field(default_factory=list)
    heuristics: List[str] = field(default_factory=list)
    predictions: List[str] = field(default_factory=list)

@dataclass
class AgentSpec:
    task: Task
    system_model: Optional[SystemModel] = None