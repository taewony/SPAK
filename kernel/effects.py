from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from .semantic_kernel import Effect

# --- LLM Effects ---
@dataclass
class LLMRequest:
    messages: List[Dict[str, str]]
    model: Optional[str] = None
    stop: Optional[List[str]] = None
    system_model: Optional[str] = None

@dataclass
class Generate(Effect[str]):
    """Effect to request text generation from an LLM."""
    payload: LLMRequest

# --- REPL Effects ---
@dataclass
class CodeExecution:
    code: str
    timeout: int = 5

@dataclass
class ExecuteCode(Effect[str]):
    """Effect to execute code in a sandbox."""
    payload: CodeExecution

# --- File System Effects ---
@dataclass
class FileRead:
    path: str

@dataclass
class ReadFile(Effect[str]):
    payload: FileRead

@dataclass
class FileList:
    dir_path: str

@dataclass
class ListFiles(Effect[List[str]]):
    payload: FileList

@dataclass
class FileWrite:
    path: str
    content: str

@dataclass
class WriteFile(Effect[None]):
    payload: FileWrite

@dataclass
class GrepRequest:
    pattern: str
    dir_path: str
    recursive: bool = True
    file_pattern: Optional[str] = None # e.g. "*.md" or "*.{md,txt}"

@dataclass
class GrepFiles(Effect[List[str]]):
    """Effect to search for a pattern in files using OS tools (grep/findstr)."""
    payload: GrepRequest

# --- Math Effects (Level 2) ---
@dataclass
class MathOperation:
    op: str # "add", "sub", "mul", "div"
    a: float
    b: float

@dataclass
class Math(Effect[float]):
    payload: MathOperation

# --- User Interaction Effects (Level 3) ---
@dataclass
class UserInput:
    prompt: Optional[str] = None

@dataclass
class Listen(Effect[str]):
    payload: UserInput

@dataclass
class UserOutput:
    message: str

@dataclass
class Reply(Effect[str]):
    payload: UserOutput

# --- Message Bus Effects (Level 4) ---
@dataclass
class Message:
    recipient: Optional[str]
    content: str
    broadcast: bool = False

@dataclass
class SendMessage(Effect[str]):
    payload: Message

# --- Agent Control Effects ---
@dataclass
class SubTask:
    query: str
    context: str
    spec_path: Optional[str] = None

@dataclass
class Recurse(Effect[str]):
    """Effect to spawn a recursive sub-agent."""
    payload: SubTask

# --- Verification Effects ---
@dataclass
class TraceLog:
    thought: str
    plan: Dict[str, Any]
    raw_response: Optional[str] = None

@dataclass
class ReasoningTrace(Effect[None]):
    """Effect to log the internal reasoning/planning step of the agent."""
    payload: TraceLog
