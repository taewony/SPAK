from typing import Any, List, Dict
from dataclasses import dataclass
from semantic_kernel import Agent, Effect, Handler

class Runtime:
    """
    The Event Loop that executes Agents defined by AISpec.
    Matches the design-spec.md definition.
    """
    def __init__(self):
        self.handlers: List[Handler] = []
        self.trace: List[Dict] = []

    def register_handler(self, handler: Handler):
        self.handlers.append(handler)

    def step(self, agent: Agent, input_signal: Any = None):
        """
        Executes a single step or a sequence of steps for the agent.
        """
        # 1. Check Pre-invariants (if agent has them)
        if hasattr(agent.spec, 'invariants'):
            for inv in agent.spec.invariants:
                # In a real system, we'd evaluate the lambda/logic here
                pass

        # 2. Run Policy Step
        # The agent.policy() is a generator. We send the input signal.
        try:
            # If it's the first step, we just call next()
            # If we are resuming, we send input_signal (which might be the result of previous effect)
            if input_signal is None:
                effect_or_result = next(agent.policy_generator)
            else:
                effect_or_result = agent.policy_generator.send(input_signal)

            # 3. Catch Effects
            if isinstance(effect_or_result, Effect):
                # 4. Route to Handler
                result = self._resolve_effect(effect_or_result)
                
                # 5. Apply State Update (Implicit in Value Semantics, but if Agent returns new state, handle it)
                # (Recursive step happens here or in the loop)
                return result
            
            return effect_or_result

        except StopIteration as e:
            return e.value
        except Exception as e:
            raise e

    def _resolve_effect(self, effect: Effect) -> Any:
        self.trace.append({"type": "effect", "name": type(effect).__name__, "payload": effect.payload})
        for handler in reversed(self.handlers):
            try:
                return handler.handle(effect)
            except NotImplementedError:
                continue
        raise RuntimeError(f"Unhandled Effect: {effect}")
