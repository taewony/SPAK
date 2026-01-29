meta {
    name = "KnowledgeChef"
    version = "3.2.0"
    domain = "Systematic Knowledge Synthesis"
    description = "Level 4: Autonomous Research & Writing System"
}

// --- The Brain: Domain Logic for Latent Planning ---
system_model KnowledgeChefModel {
    
    // 1. Axioms
    axiom: "A claim without a source citation is a hallucination."
    axiom: "Newer sources (>2024) supersede older ones for AI topics."

    // 2. Heuristics
    heuristic: "If search returns > 50 results, Refine Query with stricter keywords."
    heuristic: "If source code is found, Prefer 'Static Analysis' over 'Text Summary'."
    heuristic: "If Draft Quality < 8.0, Iterate with Reviewer feedback."

    // 3. Predictions
    prediction: "Reading a binary file -> Decoding Error."
    prediction: "Running unverified code -> Security Exception."
}

system KnowledgeChef {

    // --- Capabilities ---

    effect Planner {
        description: "The Cognitive Engine.";
        operation decide_next_step(state: String, model: String) -> String;
    }

    component Librarian {
        description: "Materials gatherer using Hybrid Search (Grep + Keywords).";
        function gather_materials(source_path: String, topic_keywords: String) -> List<String>;
    }

    component Analyst {
        description: "Extracts insights and structures the argument.";
        function extract_insights(documents: List<String>) -> List<Insight>;
        function create_structured_outline(insights: List[Insight], intent: String, structure_type: String) -> String;
    }

    component Writer {
        description: "Drafts and refines prose.";
        function draft_content(outline: String) -> String;
        function refine_content(draft: String, feedback: String) -> String;
        function finalize_artifact(content: String, path: String) -> String;
    }

    component Reviewer {
        description: "Quality Control based on strict criteria.";
        function evaluate(draft: String, criteria: List<String>) -> EvaluationResult;
    }

    // --- The "Level 4" Workflow ---
    
    component ChiefEditor {
        description: "Orchestrates the staff via Dynamic Latent Planning.";
        
        workflow SynthesizeReport(topic: String, source_dir: String, output_path: String) {
            intent = "Create a comprehensive report on: " + topic
            
            step AutonomousLoop {
                // The Agent enters a OODA Loop (Observe-Orient-Decide-Act)
                // It uses the 'KnowledgeChefModel' heuristics to plan the next move.
                loop {
                    // 1. Plan: Local LLM decides action based on Heuristics
                    action = perform Planner.decide_next_step(history, KnowledgeChefModel)
                    
                    // 2. Act: Execute the decided component
                    perform action
                    
                    // 3. Observe: Check if goal is met
                    if (action == "DONE") break
                }
            }
        }
    }
}