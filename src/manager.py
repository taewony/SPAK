from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import json
from kernel.semantic_kernel import perform
from kernel.effects import SendMessage, Message, Generate, LLMRequest, ReasoningTrace, TraceLog

@dataclass
class Task:
    id: str
    description: str
    assignee: str
    status: str = "pending"

@dataclass
class ProjectState:
    tasks: List[Task] = field(default_factory=list)
    team: Dict[str, str] = field(default_factory=dict) # Name -> Role

class Manager:
    def __init__(self):
        self.state = ProjectState()

    def add_team_member(self, member_name: str, role: str) -> str:
        if not role:
            return "Error: Role cannot be empty"
        self.state.team[member_name] = role
        return f"Member {member_name} added as {role}"

    def create_project_plan(self, goal: str) -> List[Task]:
        """
        Uses LLM to decompose the goal into tasks based on available team members.
        """
        # 1. Think
        perform(ReasoningTrace(TraceLog(
            thought=f"Goal: '{goal}'. I need to split this into tasks for my team: {list(self.state.team.keys())}.",
            plan={"action": "plan", "goal": goal}
        )))

        # 2. Generate Plan
        team_str = json.dumps(self.state.team)
        prompt = f"""
        You are a Project Manager.
        Goal: {goal}
        Team: {team_str}
        
        Create a list of tasks. Assign each task to a specific team member.
        Output JSON list of objects: {{"id": "1", "description": "...", "assignee": "Name"}}
        """
        
        response = perform(Generate(LLMRequest(messages=[{"role": "user", "content": prompt}])))
        
        try:
            # Clean JSON
            clean_json = response.strip()
            if "```json" in clean_json:
                clean_json = clean_json.split("```json")[1].split("```")[0]
            elif "```" in clean_json:
                clean_json = clean_json.split("```")[1].split("```")[0]
            
            data = json.loads(clean_json)
            tasks = [Task(status="pending", **item) for item in data]
            self.state.tasks.extend(tasks)
            return tasks
        except Exception as e:
            print(f"Plan Generation Failed: {e}")
            return []

    def assign_task(self, task: Task) -> str:
        if task.assignee in self.state.team:
            perform(SendMessage(Message(recipient=task.assignee, content=task.description)))
            task.status = "assigned"
            return f"Task '{task.id}' delegated to {task.assignee}"
        else:
            return f"Error: Assignee {task.assignee} not in team."

    def handle_message(self, sender: str, content: str) -> str:
        """
        Callback when a worker replies.
        """
        print(f"🔔 [Manager] Received from {sender}: {content}")
        # Logic to update task status based on content would go here
        return "Ack"