import logging
from typing import Dict, Any, List
from spak_kernel.dsl.schema import AgentBlueprint, Task
from spak_kernel.core.llm_client import WorkerLLM
from spak_kernel.runtime.registry import ToolRegistry, ValidatorRegistry
from spak_kernel.runtime.tracing import TraceManager

class SpakEngine:
    def __init__(self, blueprint: AgentBlueprint, model_name: str = "qwen2.5:4b"):
        self.blueprint = blueprint
        self.llm = WorkerLLM(model=model_name)
        self.memory = []  # List[Item]
        self.current_task_name = blueprint.start_task
        self.is_terminal = False
        
        # Initialize Structured Tracer for Meta-Supervision
        self.tracer = TraceManager()
        logging.info(f"📝 Trace Log Session: {self.tracer.get_log_path()}")

    def run(self, user_goal: str):
        """The Main Cognitive Loop"""
        logging.info(f"🚀 Starting Agent: {self.blueprint.name}")
        
        # Initial Context Injection
        self.memory.append({"role": "user", "content": f"Main Goal: {user_goal}"})

        while not self.is_terminal:
            current_task = self.blueprint.tasks[self.current_task_name]
            logging.info(f"⚡ Entering Task: {self.current_task_name}")

            # 1. Context Builder (Stateless Worker Prep)
            # System Model Axioms + Current Task Goal + Summarized Memory
            prompt = self._build_prompt(current_task)

            # 2. Plan (Worker LLM)
            # Returns strictly parsed ADT Items (Reasoning + FunctionCalls)
            # Captured raw_text for tracing
            raw_response, response_items = self.llm.generate_with_raw(prompt)
            
            # 3. Act (Tool Execution)
            execution_results = []
            tool_logs = []
            
            for item in response_items:
                if item.type == 'function_call':
                    logging.info(f"🛠️ Tool Call: {item.name}({item.args})")
                    result = ToolRegistry.execute(item.name, item.args)
                    execution_results.append(result)
                    
                    tool_logs.append({
                        "tool": item.name,
                        "args": item.args,
                        "result": str(result)[:500] # Truncate for log size
                    })
            
            # 4. Observe (Validator)
            # Check if the execution results satisfy the task requirement
            validator = ValidatorRegistry.get(current_task.validator)
            trigger_signal = validator.validate(execution_results, current_task)
            
            logging.info(f"🔍 Validator Signal: {trigger_signal}")

            # 5. Trace (Record the Cognitive Step)
            self.tracer.log_step(
                task=current_task,
                prompt_context=prompt,
                llm_response_raw=raw_response,
                parsed_items=response_items,
                tool_executions=tool_logs,
                validator_signal=trigger_signal
            )

            # 6. Refine (State Transition)
            next_task = current_task.transitions.get(trigger_signal)
            
            if next_task == "END":
                self.is_terminal = True
                logging.info("✅ Agent Mission Completed.")
            elif next_task:
                # Flush memory if needed based on Config
                if self.blueprint.config.get("MEMORY") == "FLUSH_ON_TRANSITION":
                    self._summarize_and_flush_memory()
                self.current_task_name = next_task
            else:
                # Heuristic Fallback (Self-Correction Logic)
                logging.warning(f"⚠️ Trigger '{trigger_signal}' has no transition. Applying Heuristics.")
                self._inject_heuristic_guidance()
    
    def _build_prompt(self, task: Task) -> list:
        """Constructs the exact context for the Worker LLM"""
        system_msg = self.blueprint.system_model_prompt() # Axioms
        task_msg = f"Current Phase: {task.name}\nGoal: {task.goal}\nAvailable Tools: {task.tools}"
        return [
            {"role": "system", "content": system_msg},
            {"role": "system", "content": task_msg},
            *self.memory # Or filtered memory
        ]