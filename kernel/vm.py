import json
import os
import time
from typing import Dict, Any, Optional
from .dsl.ast import Task, Step, ToolStep, LLMStep, Metric

class ExecutionContext:
    def __init__(self):
        self.variables: Dict[str, Any] = {}
        self.step_pointer: int = 0
        self.trace_log: list = []
        self.metrics: Dict[str, Any] = {}
        self.status: str = "READY" # READY, RUNNING, SUSPENDED, COMPLETED, FAILED
        self.dsl_path: str = ""

    def update(self, key: str, value: Any):
        if key:
            self.variables[key] = value

    def to_dict(self):
        return {
            "variables": self.variables,
            "step_pointer": self.step_pointer,
            "trace_log": self.trace_log,
            "metrics": self.metrics,
            "status": self.status,
            "dsl_path": self.dsl_path
        }

    def load_dict(self, data: dict):
        self.variables = data.get("variables", {})
        self.step_pointer = data.get("step_pointer", 0)
        self.trace_log = data.get("trace_log", [])
        self.metrics = data.get("metrics", {})
        self.status = data.get("status", "READY")
        self.dsl_path = data.get("dsl_path", "")

class StepMachine:
    def __init__(self, task: Task, resume_state_path: Optional[str] = None, dsl_path: str = "", backend: str = "sim", live_mode: bool = False):
        self.task = task
        self.context = ExecutionContext()
        self.context.dsl_path = dsl_path
        self.resume_path = resume_state_path
        self.backend = backend
        self.live_mode = live_mode
        
        if dsl_path:
            self.output_dir = os.path.dirname(os.path.abspath(dsl_path))
        else:
            self.output_dir = os.getcwd()
        
        # Load state if resuming
        if resume_state_path and os.path.exists(resume_state_path):
            with open(resume_state_path, "r") as f:
                self.context.load_dict(json.load(f))
                print(f"[*] Resumed from step {self.context.step_pointer}")

    def _call_ollama(self, prompt: str, model: str = "qwen2.5:7b") -> str:
        import urllib.request
        import json
        
        url = "http://localhost:11434/api/generate"
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        
        req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result.get("response", "").strip().replace("\r\n", "\n")
        except Exception as e:
            print(f"[!] Ollama Call Failed: {e}")
            return f"Error: {e}"

    def log_step(self, step_id: str, step_type: str, details: dict):
        entry = {
            "timestamp": time.time(),
            "step_id": step_id,
            "type": step_type,
            "details": details
        }
        self.context.trace_log.append(entry)
        self._save_trace()

    def _save_trace(self):
        # Save trace.json
        trace_path = os.path.join(self.output_dir, "trace.json")
        with open(trace_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(self.context.trace_log, f, indent=2)
        
        # Save context snapshot for resume
        context_path = os.path.join(self.output_dir, "context.json")
        with open(context_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(self.context.to_dict(), f, indent=2)
            
        # Save metrics.json
        metrics_path = os.path.join(self.output_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(self.context.metrics, f, indent=2)

    def render_prompt(self, template: str) -> str:
        # Simple jinja-like variable substitution
        prompt = template
        for key, val in self.context.variables.items():
            prompt = prompt.replace(f"{{{{{key}}}}}", str(val))
        
        # Access system model info
        if self.task.system_model:
            for i, val in enumerate(self.task.system_model.axioms):
                prompt = prompt.replace(f"{{{{axiom.{i}}}}}", val)
            for i, val in enumerate(self.task.system_model.heuristics):
                prompt = prompt.replace(f"{{{{heuristic.{i}}}}}", val)
        
        return prompt

    def _execute_step_logic(self, step: Step, step_id: str) -> bool:
        # Returns True if step completed, False if suspended
        
        if isinstance(step, ToolStep):
            # --- Tool: tool.run (Shell) ---
            if step.tool_name == "run":
                cmd = step.params.get('cmd', '')
                cmd_expanded = self.render_prompt(cmd)
                
                print(f"    > Running Tool: {step.tool_name} args={cmd_expanded}")
                
                if self.live_mode:
                    import subprocess
                    print("    [LIVE EXECUTION] Spawning subprocess...")
                    try:
                        # Use shell=True for flexibility with pipes/echo, though simpler is safer.
                        # We execute in output_dir to ensure file paths work.
                        proc = subprocess.run(
                            cmd_expanded, 
                            shell=True, 
                            capture_output=True, 
                            text=True, 
                            cwd=self.output_dir
                        )
                        result = proc.stdout.strip()
                        if proc.stderr:
                            result += f"\n[STDERR]: {proc.stderr.strip()}"
                        print(f"    [LIVE RESULT] Exit Code: {proc.returncode}")
                    except Exception as e:
                        result = f"EXECUTION ERROR: {e}"
                else:
                    # Mock execution
                    result = f"(Mock Output of {cmd_expanded})"
                    if "echo" in cmd_expanded:
                         import re
                         # Handle quoted strings: echo "hello" or echo 'hello'
                         m = re.search(r"echo\s+(['\"])(.*)\1", cmd_expanded)
                         if m:
                             result = m.group(2)
                         else:
                             # Handle unquoted: echo 10
                             result = cmd_expanded.replace("echo ", "").strip()

                if step.output_var:
                    self.context.update(step.output_var, result)
                
                self.log_step(step_id, step.type, {"cmd": cmd_expanded, "output": result})
                return True

            # --- Tool: tool.write (File Write) ---
            elif step.tool_name == "write":
                path = step.params.get('path', '')
                content = step.params.get('content', '')
                
                # Resolve relative path
                if not os.path.isabs(path):
                    path = os.path.join(self.output_dir, path)
                
                content_expanded = self.render_prompt(content)
                
                print(f"    > Writing File: {path}")
                try:
                    with open(path, "w", encoding="utf-8", newline="\n") as f:
                        f.write(content_expanded)
                    result = "SUCCESS"
                except Exception as e:
                    result = f"ERROR: {e}"
                
                if step.output_var:
                    self.context.update(step.output_var, result)
                self.log_step(step_id, step.type, {"path": path, "status": result})
                return True

            # --- Tool: tool.read (File Read) ---
            elif step.tool_name == "read":
                path = step.params.get('path', '')
                if not os.path.isabs(path):
                    path = os.path.join(self.output_dir, path)
                
                print(f"    > Reading File: {path}")
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        result = f.read()
                except Exception as e:
                    result = f"ERROR: {e}"
                
                if step.output_var:
                    self.context.update(step.output_var, result)
                self.log_step(step_id, step.type, {"path": path})
                return True

        elif isinstance(step, LLMStep):
            prompt = self.render_prompt(step.prompt_template)
            
            # --- OLLAMA BACKEND ---
            if self.backend == "ollama":
                print(f"    > Calling Ollama (qwen2.5:7b)...")
                response_content = self._call_ollama(prompt)
                print(f"    -> Response: {response_content[:100]}...") 
                
                if step.output_var:
                     self.context.update(step.output_var, response_content)
                
                self.log_step(step_id, step.type, {"prompt": prompt, "response": response_content})
                return True

            # --- SIMULATION BACKEND ---
            response_file = os.path.join(self.output_dir, "response.txt")
            
            if os.path.exists(response_file):
                 print(f"[*] Found response in {response_file}. Resuming...")
                 with open(response_file, "r", encoding="utf-8") as f:
                     response_content = f.read().strip().replace("\r\n", "\n")
                 
                 if not response_content:
                     print("[!] response.txt is empty. Suspending again.")
                     self._suspend_for_input(step_id, step.role, prompt)
                     return False

                 if step.output_var:
                     self.context.update(step.output_var, response_content)
                 
                 self.log_step(step_id, step.type, {"prompt": prompt, "response": response_content})
                 
                 # Cleanup
                 os.remove(response_file)
                 return True

            else:
                self._suspend_for_input(step_id, step.role, prompt)
                return False
        
        return True

    def run(self):
        self.context.status = "RUNNING"
        steps = self.task.steps
        
        while self.context.step_pointer < len(steps):
            step = steps[self.context.step_pointer]
            print(f"[{self.context.step_pointer}] Executing {step.id} ({step.type})...")

            completed = self._execute_step_logic(step, step.id)
            if not completed:
                return # Suspended

            self.context.step_pointer += 1
            self._save_trace()

        print("[*] Task Completed. Running Evaluation...")
        self.run_evaluation()
        self.context.status = "COMPLETED"
        self._save_trace()

    def run_evaluation(self):
        print("\n--- [Evaluation Phase] ---")
        if not self.task.evaluation:
            print("No evaluation metrics defined.")
            return

        for metric in self.task.evaluation:
            if metric.id in self.context.metrics:
                print(f"[*] Skipping Metric: {metric.id} (Already Evaluated)")
                continue

            print(f"[*] Checking Metric: {metric.id}...")
            # We execute the inner logic step
            # Note: Metrics don't advance the main step_pointer, but we might want to log them?
            # We treat them as atomic for now (no suspend/resume support inside evaluation yet? 
            # Actually, if an evaluation needs LLM, it MIGHT suspend. 
            # This complicates things. For now, assume evaluation LLM is also simulated via response.txt?
            # Or assume evaluation is usually automated?
            # User wants to test effectiveness, so likely LLM-as-a-judge.
            # If it suspends, we need to handle that.
            
            # Current VM architecture relies on step_pointer for the main loop.
            # We need a separate state for evaluation or just execute linearly?
            # If we suspend during evaluation, we need to know WHICH metric we are on.
            
            # Simple Hack: Just execute. If it returns False (suspended), we stop.
            # But resuming needs to know we are in evaluation mode.
            # Let's skip suspend support for evaluation for this POC, OR 
            # force user to provide all responses?
            # Let's just try to run it. If it suspends, the user needs to provide response and we resume.
            # BUT resume logic in 'run' starts from step_pointer. 
            # It doesn't know about evaluation pointer.
            
            # For this step, I will only implement Tool-based metrics or LLM metrics that use a different mechanism?
            # No, user wants LLM.
            
            # Let's implement 'evaluation' as appended steps? No.
            # Let's just run them. If one suspends, we are stuck.
            # Okay, I will implement basic tool-based evaluation only for now to ensure safety.
            # Or better: Add a simple blocking input for evaluation if needed? 
            # "Simulating LLM Judge: Enter score..."
            
            pass
            
            # Execute logic
            # We need to capture the output into metrics dict, not just variables.
            # The inner step has output_var.
            
            # To support output, we might need to trap the variable update.
            # Let's just run it. The output goes to context.variables.
            # Then we copy it to context.metrics[metric.id].
            
            completed = self._execute_step_logic(metric.logic, metric.id)
            if completed:
                # Assuming output_var holds the result
                if metric.logic.output_var:
                    val = self.context.variables.get(metric.logic.output_var)
                    self.context.metrics[metric.id] = val
                    print(f"    -> Result: {val}")
            else:
                print(f"[!] Evaluation Metric {metric.id} suspended. (Resume not supported in Eval yet)")

    def _suspend_for_input(self, step_id: str, role: str, prompt: str):
        print(f"\n--- [LLM Interaction Required] ---")
        print(f"Step: {step_id}")
        print(f"Role: {role}")
        print(f"Prompt:\n{prompt}\n")
        print("-" * 30)
        print(f"[*] System Suspended. Please write your response to 'response.txt' and run:")
        print(f"    spak step --resume")
        
        self.context.status = "SUSPENDED_WAITING_INPUT"
        self._save_trace()
        # Create empty response file if not exists
        response_file = os.path.join(self.output_dir, "response.txt")
        if not os.path.exists(response_file):
            with open(response_file, "w") as f:
                f.write("")
