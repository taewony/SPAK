from typing import List, Dict
from src.librarian import Librarian
from src.analyst import Analyst
from src.writer import Writer

class ResearchAgent:
    """
    The composite agent class for ResearchAnalyst system.
    Aggregates the Librarian, Analyst, and Writer components.
    State is managed here.
    """
    def __init__(self):
        self.librarian = Librarian()
        self.analyst = Analyst()
        self.writer = Writer()
        
        # Agent State
        self.state = {
            "raw_docs": [],
            "insights": "",
            "outline": "",
            "draft": "",
            "status": "idle"
        }

    def gather_info(self, topic: str, path: str = "docs") -> str:
        """Step 1: Gather information."""
        print(f"🔎 [Agent] Gathering info on '{topic}' from '{path}'...")
        docs = self.librarian.gather_materials(path, topic)
        self.state["raw_docs"] = docs
        self.state["status"] = "gathered"
        return f"Found {len(docs)} documents."

    def analyze_docs(self) -> str:
        """Step 2: Analyze gathered documents."""
        if not self.state["raw_docs"]:
            return "Error: No documents to analyze. Run gather_info first."
        
        print(f"🧠 [Agent] Analyzing {len(self.state['raw_docs'])} docs...")
        insights = self.analyst.extract_insights(self.state["raw_docs"])
        self.state["insights"] = insights
        self.state["status"] = "analyzed"
        return f"Insights extracted: {insights[:100]}..."

    def create_outline(self, intent: str) -> str:
        """Step 3: Create an outline."""
        if not self.state["insights"]:
            return "Error: No insights available. Run analyze_docs first."
            
        print(f"📝 [Agent] Creating outline for intent: {intent}...")
        outline = self.analyst.map_to_storyline(
            self.state["insights"], 
            intent, 
            "Academic Paper Structure"
        )
        self.state["outline"] = outline
        self.state["status"] = "outlined"
        return f"Outline created: {outline[:100]}..."

    def write_paper(self, filename: str) -> str:
        """Step 4: Write the final paper."""
        if not self.state["outline"]:
            return "Error: No outline available. Run create_outline first."
            
        print(f"✍️ [Agent] Drafting content...")
        draft = self.writer.draft_content(self.state["outline"], "Markdown")
        self.state["draft"] = draft
        
        print(f"💾 [Agent] Saving to {filename}...")
        result = self.writer.finalize_artifact(draft, filename)
        self.state["status"] = "completed"
        return result
