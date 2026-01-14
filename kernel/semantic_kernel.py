from typing import Any, Callable, Dict, List, Optional, TypeVar, Generic
from dataclasses import dataclass, replace, field
from abc import ABC, abstractmethod
import inspect

# --- 1. Value Semantics (State) ---

@dataclass(frozen=True)
class State:
    """Base class for all Agent States. Immutable."""
    pass

# --- 2. Algebraic Effects ---

T = TypeVar("T")

@dataclass
class Effect(Generic[T]):
    """
    Represents a side-effect to be performed.
    It carries the intent, but not the execution logic.
    """
    payload: Any

class EffectRequest(Exception):
    """
    Raised to pause execution and request an effect handling.
    This mimics Algebraic Effects in Python (via Exception bubbling).
    """
    def __init__(self, effect: Effect, callback: Callable[[Any], Any]):
        self.effect = effect
        self.callback = callback

def perform(effect: Effect[T]) -> T:
    """
    Perform an effect. This pauses the current function 
    and yields control to the nearest Handler.
    """
    # In a generator-based implementation, this would be `yield effect`
    # In a stack-based implementation, we can use exceptions or coroutines.
    # Here we simulate via a primitive continuation passing style hook or exception.
    # For simplicity in this prototype, we'll use a specific Exception control flow.
    result = None
    
    def resume(val):
        nonlocal result
        result = val
        
    raise EffectRequest(effect, resume)
    return result # type: ignore

# --- 3. Handlers ---

class Handler(ABC):
    """
    Intercepts effects and resolves them.
    """
    @abstractmethod
    def handle(self, effect: Effect) -> Any:
        pass

# --- 4. Agent Definition (The Semantic Unit) ---

@dataclass
class AgentSpec:
    name: str
    description: str
    invariants: List[Callable[[State], bool]] = field(default_factory=list)

class Agent(ABC):
    def __init__(self, spec: AgentSpec, initial_state: State):
        self.spec = spec
        self.state = initial_state

    @abstractmethod
    def policy(self) -> Any:
        """
        The logic loop. 
        MUST be a generator or coroutine to allow pausing for Effects.
        """
        pass

# --- 5. The Runtime (Kernel) ---

class Runtime:
    def __init__(self):
        self.handlers: List[Handler] = []
        self.trace: List[Dict] = []

    def register_handler(self, handler: Handler):
        self.handlers.append(handler)

    def run(self, agent: Agent, max_steps: int = 100):
        """
        Executes the agent policy, handling effects via the stack.
        """
        generator = agent.policy()
        last_result = None
        
        step = 0
        while step < max_steps:
            try:
                # check pre-invariants
                if not all(inv(agent.state) for inv in agent.spec.invariants):
                    raise RuntimeError(f"Invariant failed for {agent.spec.name}")

                # Resume execution
                future = generator.send(last_result) if step > 0 else next(generator)
                
                # If the policy yields an Effect directly (alternative to Exception)
                if isinstance(future, Effect):
                    last_result = self._resolve_effect(future)
                else:
                    # Policy yielded a new State?
                    if isinstance(future, State):
                        agent.state = future
                        last_result = None
                    else:
                        last_result = future

            except StopIteration as e:
                return e.value
            except Exception as e:
                raise e
            
            step += 1

    def _resolve_effect(self, effect: Effect) -> Any:
        self.trace.append({"type": "effect", "name": type(effect).__name__, "payload": effect.payload})
        
        # Walk stack backwards
        for handler in reversed(self.handlers):
            try:
                return handler.handle(effect)
            except NotImplementedError:
                continue
        
        raise RuntimeError(f"Unhandled Effect: {effect}")
