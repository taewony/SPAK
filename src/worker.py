from kernel.semantic_kernel import perform
from kernel.effects import SendMessage, Message, Generate, LLMRequest

class Worker:
    def do_work(self, task_desc: str) -> str:
        if not task_desc:
            return "Error: Empty task"
        
        print(f"👷 [Worker] Working on: {task_desc}...")
        
        # Simulate work using LLM
        prompt = f"Execute the following task concisely: {task_desc}"
        result = perform(Generate(LLMRequest(messages=[{"role": "user", "content": prompt}])))
        
        # Report back
        perform(SendMessage(Message(recipient="Manager", content=f"Finished: {task_desc}. Result: {result[:50]}...")))
        
        return result