meta {
    name = "ProjectTeam"
    version = "2.0.0"
    domain = "Multi-Agent Collaboration"
    purpose = "To coordinate a team of agents via Message Bus to execute complex projects."
    description = "Level 4: Choreography Pattern (Manager <-> Worker)"
}

contract AgentScope {
    supported_intents = [
        "Task Delegation",
        "Project Management"
    ]
    success_criteria = [
        "Message Delivery: All tasks reach correct workers.",
        "Loop Closure: Workers report completion back to Manager."
    ]
}

system ProjectTeam {

    // --- Domain-IR ---
    struct Task {
        id: String
        description: String
        assignee: String
        status: String
    }

    struct ProjectStatus {
        goal: String
        active_tasks: Number
        completed_tasks: Number
    }

    // --- Effects ---
    effect MessageBus {
        // Asynchronous communication
        operation send(recipient: String, content: String) -> String;
        operation broadcast(content: String) -> String;
    }

    effect CognitiveEngine {
        operation plan(goal: String) -> List[Task];
    }

    // --- Components ---

    component Manager {
        description: "The Coordinator. Decomposes goals into tasks and tracks status.";

        state ProjectState {
            tasks: List[Task]
            team: Map[String, String] // Name -> Role
        }

        function add_team_member(name: String, role: String) -> String;
        
        function create_project_plan(goal: String) -> List[Task];
        
        function assign_task(task: Task) -> String;
        
        // Handling incoming messages (e.g., "Finished Task X")
        function handle_message(sender: String, content: String) -> String;
    }

    component Worker {
        description: "The Executor. Performs tasks and reports back.";

        function do_work(task_desc: String) -> String;
    }

    // --- Workflows ---

    workflow ExecuteProject(goal: String) {
        step Plan {
            // Manager thinks and creates a plan
            tasks = perform Manager.create_project_plan(goal)
        }

        step Delegate {
            // Loop through tasks and assign
            // (Simplified representation)
            perform Manager.assign_task(tasks[0]) 
        }

        step WaitAndMonitor {
            // In a real async system, this would be an event loop
            // Here we imply the Manager waits for callbacks
        }
    }
}