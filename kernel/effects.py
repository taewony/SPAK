from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from .semantic_kernel import Effect

# --- LLM Effects ---
@dataclass
class LLMRequest:
    messages: List[Dict[str, str]]
    model: Optional[str] = None
    stop: Optional[List[str]] = None

@dataclass
class Generate(Effect[str]):
    """Effect to request text generation from an LLM."""
    payload: LLMRequest

# --- REPL Effects ---
@dataclass
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
class FileWrite:
    path: str
    content: str

@dataclass
class WriteFile(Effect[None]):
    payload: FileWrite

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