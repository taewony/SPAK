meta {
    name = "ResearchAnalyst"
    version = "2.0.0"
    domain = "Systematic Knowledge Synthesis"
    description = "The Knowledge Chef: A sophisticated agent capable of synthesizing raw documents into structured papers with claim-evidence mapping."
}

contract AgentScope {
    supported_intents = [
        "Research Synthesis",
        "Artifact Creation"
    ]
    
    success_criteria = [
        "Source Fidelity: All claims must be backed by evidence.",
        "Structural Integrity: Output must strictly follow the requested template."
    ]
}

system ResearchAnalyst {

    // --- Domain-IR (Structured Data) ---
    struct Insight {
        claim: String
        evidence: String
        source: String
        confidence: Number
    }

    struct EvaluationResult {
        status: String
        feedback: String
        score: Number
    }

    // --- Effects ---
    effect FileSystem {
        operation read_file(path: String) -> String;
        operation list_files(dir_path: String) -> List<String>;
        operation grep_files(pattern: String, dir_path: String) -> List<String>;
        operation write_file(path: String, content: String) -> String;
    }

    effect LLM {
        operation generate(prompt: String) -> String;
    }

    // --- Components ---
    
    component ChiefEditor {
        description: "The Orchestrator. Coordinates the entire paper authoring workflow.";
        function start() -> String;
    }

    component Librarian {
        description: "The Buyer. Hybrid engine engaging fast tools (grep) for speed and LLMs for depth.";
        function gather_materials(source_path: String, topic_keywords: String) -> List<String>;
    }

    component Analyst {
        description: "The Cook. Extracts strictly typed insights and maps them to a storyline.";
        function extract_insights(documents: List<String>) -> List<Insight>;
        function create_structured_outline(insights: List<Insight>, intent: String, structure_type: String) -> String;
    }

    component Writer {
        description: "The Plater. Drafts content and refines it based on feedback.";
        function draft_content(outline: String) -> String;
        function refine_content(draft: String, feedback: String) -> String;
        function finalize_artifact(content: String, path: String) -> String;
    }

    component Reviewer {
        description: "The Taster. Evaluates artifacts against success criteria.";
        
        // --- Domain Invariants ---
        invariant: "Final artifact must contain all required sections (Motivation, Background, Methodology, Results)"
        invariant: "No hallucinated citations allowed (Result coverage > 0.9)"
        
        function evaluate(draft: String, criteria: List<String>) -> EvaluationResult;
    }

    // --- Workflows (Source of Operational Consistency Plan) ---

    workflow AuthorPaper(source_dir: String, intent: String, structure_type: String) {
        step SmartCollection {
            docs = perform Librarian.gather_materials(source_dir, intent)
        }
        step InsightExtraction {
            insights = perform Analyst.extract_insights(docs)
        }
        step StructuralBlueprinting {
            outline = perform Analyst.create_structured_outline(insights, intent, structure_type)
        }
        step Drafting {
            draft = perform Writer.draft_content(outline)
        }
        step QualityControl {
            eval = perform Reviewer.evaluate(draft, ["Source Fidelity", "Logical Flow"])
        }
        step Finalization {
            perform Writer.finalize_artifact(draft, "final_paper.md")
        }
    }
}
