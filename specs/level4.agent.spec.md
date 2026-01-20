meta {
    name = "ProjectTeam"
    version = "1.0"
    domain = "Multi-Agent Collaboration"
    purpose = "To coordinate a team of agents to execute complex projects."
    description = "Level 4: Multi-Agent System with Manager and Worker"
}

// --- Operational Contract ---
contract AgentScope {
    supported_intents = [
        "Task Delegation: Assigning work to workers.",
        "Project Management: Tracking status and goals."
    ]

    success_criteria = [
        "Coordination: Messages are correctly routed.",
        "Task Completion: Workers successfully report back results."
    ]

    autonomy {
        mode = "autonomous"
        loop_interval = "10s"
        max_steps = 100
    }
}

system ProjectTeam {
    effect MessageBus {
        operation send(recipient: String, content: String) -> String;
        operation broadcast(content: String) -> String;
    }

    component Manager {
        description: "Decomposes tasks and delegates them to workers.";

        state ProjectState {
            task_queue: List[String]
            status: String
        }

        workflow ExecuteProject(goal: String) {
            step PlanTask {
                # In a real implementation, this would involve LLM thinking
                perform MessageBus.broadcast("New Project Goal: " + goal)
            }

            step Delegate {
                perform MessageBus.send("Worker", "Please execute task: Research")
            }
        }
    }

    component Worker {
        description: "Executes tasks assigned by the Manager.";

        function do_work(task: String) -> String {
            return "Completed task: " + task
        }
    }
}
