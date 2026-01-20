meta {
    name = "ResearchAnalyst"
    version = "0.1.0"
    domain = "Knowledge Synthesis & Research"
    purpose = "To autonomously gather, analyze, and synthesize raw information into structured documents."
    description = "A sophisticated agent capable of synthesizing raw documents into structured papers, reports, and educational materials with claim-evidence mapping."
}

// --- Operational Contract ---
// Explicitly defining the scope and success metrics for this agent.
contract AgentScope {
    supported_intents = [
        "Research Synthesis: Aggregating multiple sources into a summary.",
        "Artifact Creation: Drafting papers, reports, or courseware.",
        "Evidence Mapping: Tracing claims back to source documents."
    ]

    success_criteria = [
        "Source Fidelity: All claims must be backed by evidence.",
        "Structural Integrity: Output must strictly follow the requested template.",
        "Completeness: Key insights from the source set are not omitted."
    ]

    autonomy {
        mode = "autonomous"
        loop_interval = "5s"
        max_steps = 50
    }
}

system ResearchAnalyst {

    // --- 1. Effects (Capabilities) ---
    effect FileSystem {
        operation read_file(path: String) -> String;
        operation list_files(dir_path: String) -> List<String>;
        operation write_file(path: String, content: String) -> String;
    }

    effect LLM {
        operation generate(prompt: String) -> String;
    }

    // --- 2. Components (Roles) ---
    
    component Librarian {
        description: "Responsible for data ingestion. Scans directories and filters relevant documents based on the topic.";
        
        function gather_materials(source_path: String, topic_keywords: String) -> List<String>;
    }

    component Analyst {
        description: "The logic core. Extracts key claims, evidences, and organizes them into a coherent knowledge map.";

        function extract_insights(documents: List<String>) -> String;
        
        function map_to_storyline(insights: String, user_intent: String, storyline_template: String) -> String;
    }

    component Writer {
        description: "The creative output engine. Drafts the actual content in requested formats (Markdown, HTML, etc.).";

        function draft_content(outline: String, format_style: String) -> String;
        
        function finalize_artifact(draft: String, file_path: String) -> String;
    }

    // --- 3. Workflows (Scenarios) ---

    // Level 1: Simple Summary
    workflow SummarizeResearch(source_dir: String, topic: String) {
        step Collect {
            raw_docs = perform Librarian.gather_materials(source_dir, topic)
        }
        step Analyze {
            insights = perform Analyst.extract_insights(raw_docs)
        }
        step Report {
            summary = perform Writer.draft_content(insights, "Academic Summary")
            perform Writer.finalize_artifact(summary, "research_summary.md")
        }
    }

    // Level 2: Advanced Authoring (Paper/Courseware)
    workflow AuthorPaper(source_dir: String, intent: String, storyline: String) {
        step Preparation {
            raw_docs = perform Librarian.gather_materials(source_dir, intent)
            insights = perform Analyst.extract_insights(raw_docs)
        }
        step Structuring {
            structured_outline = perform Analyst.map_to_storyline(insights, intent, storyline)
        }
        step Drafting {
            paper_draft = perform Writer.draft_content(structured_outline, "Academic Paper")
            perform Writer.finalize_artifact(paper_draft, "final_paper.md")
        }
    }
}
