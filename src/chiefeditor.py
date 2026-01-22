from kernel.semantic_kernel import perform
from kernel.effects import ReasoningTrace, TraceLog
from src.librarian import Librarian
from src.analyst import Analyst
from src.writer import Writer
from src.reviewer import Reviewer

class ChiefEditor:
    def __init__(self, source_dir="docs", topic="Systematic Intelligence Engineering", structure="Academic Paper"):
        self.source_dir = source_dir
        self.topic = topic
        self.structure = structure
        
        # Instantiate sub-agents
        self.librarian = Librarian()
        self.analyst = Analyst()
        self.writer = Writer()
        self.reviewer = Reviewer()

    def start(self):
        print(f"🎬 [ChiefEditor] Starting 'AuthorPaper' workflow...")
        print(f"   Source: {self.source_dir}")
        print(f"   Topic: {self.topic}")
        
        # 1. SmartCollection
        perform(ReasoningTrace(payload=TraceLog(
            thought=f"I need to gather materials about '{self.topic}' from '{self.source_dir}'.",
            plan={"action": "gather_materials"}
        )))
        docs = self.librarian.gather_materials(self.source_dir, self.topic)
        print(f"   ✅ Collected {len(docs)} documents.")

        # 2. InsightExtraction
        perform(ReasoningTrace(payload=TraceLog(
            thought="Now I will extract insights from the collected documents.",
            plan={"action": "extract_insights"}
        )))
        insights = self.analyst.extract_insights(docs)
        print(f"   ✅ Extracted {len(insights)} insights.")

        # 3. StructuralBlueprinting
        perform(ReasoningTrace(payload=TraceLog(
            thought=f"Creating a structured outline for a '{self.structure}'.",
            plan={"action": "create_structured_outline"}
        )))
        outline = self.analyst.create_structured_outline(insights, self.topic, self.structure)
        print(f"   ✅ Outline created.")

        # 4. Drafting
        perform(ReasoningTrace(payload=TraceLog(
            thought="Drafting the full content based on the outline.",
            plan={"action": "draft_content"}
        )))
        draft = self.writer.draft_content(outline)
        print(f"   ✅ Draft generated ({len(draft)} chars).")

        # 5. QualityControl
        perform(ReasoningTrace(payload=TraceLog(
            thought="Evaluating the draft against quality criteria.",
            plan={"action": "evaluate"}
        )))
        eval_res = self.reviewer.evaluate(draft, ["Source Fidelity", "Logical Flow"])
        print(f"   ✅ Evaluation: {eval_res.score}/10 - {eval_res.status}")

        # 6. Finalization
        perform(ReasoningTrace(payload=TraceLog(
            thought="Finalizing and saving the artifact.",
            plan={"action": "finalize_artifact"}
        )))
        path = self.writer.finalize_artifact(draft, "final_paper.md")
        print(f"🎉 [ChiefEditor] Paper published to: {path}")
        
        return "Workflow Complete"
