from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid

class ItemType(Enum):
    REASONING = "reasoning"
    FUNCTION_CALL = "function_call"
    MESSAGE = "message"
    HYPOTHESIS = "hypothesis"      # New
    VALIDATION_PLAN = "validation_plan" # New
    CONSTRAINT = "constraint"      # New
    OBSERVATION = "observation"    # New

@dataclass
class OpenResponseItem:
    """Worker-LLM과 주고받는 기본 통신 단위"""
    type: ItemType
    content: Dict[str, Any]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }

# (기존 Task, AgentBlueprint 클래스도 이곳에 유지)
@dataclass
class Task:
    name: str
    goal: str
    tools: List[str]
    validator: Optional[str] = None
    transitions: Dict[str, str] = field(default_factory=dict)

@dataclass
class AgentBlueprint:
    name: str
    level: int
    config: Dict[str, Any]
    start_task: str
    tasks: Dict[str, Task]
    system_model_prompts: List[str] = field(default_factory=list)