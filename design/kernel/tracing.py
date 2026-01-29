import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
from spak_kernel.dsl.schema import Task, OpenResponseItem

@dataclass
class ExecutionTrace:
    """실행 추적 데이터 스키마"""
    trace_id: str
    session_id: str
    start_time: str
    items: List[Dict] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

class TraceManager:
    def __init__(self, log_dir: str = "logs"):
        self.session_id = f"sess_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"trace_{self.session_id}.jsonl"
        
        # 현재 활성 트레이스 관리
        self.current_trace = ExecutionTrace(
            trace_id=str(uuid.uuid4()),
            session_id=self.session_id,
            start_time=datetime.now().isoformat()
        )

    def log_step(self, 
                 task: Task, 
                 context_snapshot: List[Dict],
                 raw_response: str,
                 parsed_items: List[OpenResponseItem],
                 validator_signal: str):
        """
        Records a comprehensive cognitive step.
        """
        step_record = {
            "timestamp": time.time(),
            "task": task.name,
            "context_summary": f"{len(str(context_snapshot))} chars",
            "llm_raw": raw_response,
            "parsed_items": [item.to_dict() for item in parsed_items],
            "validator_signal": validator_signal
        }
        
        # In-memory update
        self.current_trace.items.append(step_record)

        # Append to JSONL File
        entry = {
            "trace_id": self.current_trace.trace_id,
            "session_id": self.session_id,
            "step_data": step_record
        }

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    def get_log_path(self) -> str:
        return str(self.log_file)