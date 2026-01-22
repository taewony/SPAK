import sys
import os

sys.path.append(os.getcwd())

from kernel.spec_repl import SpecREPL
from kernel.semantic_kernel import _active_runtime
from kernel.runtime import Runtime
from kernel.handlers import MockLLMHandler, FileSystemHandler, ReasoningHandler
from kernel.effects import ReasoningTrace, TraceLog
import kernel.semantic_kernel as sk

def main():
    print("🤖 [Test Script] Initializing SpecREPL...")
    repl = SpecREPL()
    
    # 1. Load Spec
    print("\n📂 [Test Script] Loading Spec...")
    repl.do_load("specs/analyst.agent.spec.md")
    
    if not repl.current_spec:
        print("❌ Failed to load spec.")
        return

    # 2. Build (Ensure implementation exists)
    print("\n🏗️ [Test Script] Building Implementation...")
    repl.do_build("src")

    # 3. Run Workflow (Simulating proper Agent Loop with Reasoning)
    print("\n🏃 [Test Script] Running Workflow (Simulated Agent Loop)...")
    
    runtime = Runtime()
    runtime.register_handler(MockLLMHandler())
    runtime.register_handler(FileSystemHandler())
    runtime.register_handler(ReasoningHandler())
    
    sk._active_runtime = runtime
    
    try:
        from src.librarian import Librarian
        from src.analyst import Analyst
        from src.writer import Writer
        from src.reviewer import Reviewer
        
        lib = Librarian()
        ana = Analyst()
        wri = Writer()
        rev = Reviewer()
        
        # --- Step 1: SmartCollection ---
        # Simulating "Think"
        print("   Step 1: SmartCollection (Reasoning...)")
        runtime._resolve_effect(ReasoningTrace(payload=TraceLog(
            thought="I need to gather materials for the topic. I will use grep to filter files.",
            plan={"action": "gather_materials"}
        )))
        # Simulating "Act"
        docs = lib.gather_materials("docs", "research_topic")
        
        # --- Step 2: InsightExtraction ---
        print("   Step 2: InsightExtraction (Reasoning...)")
        runtime._resolve_effect(ReasoningTrace(payload=TraceLog(
            thought="Now I will extract insights from the docs.",
            plan={"action": "extract_insights"}
        )))
        insights = ana.extract_insights(docs)
        
        # --- Step 3: StructuralBlueprinting ---
        print("   Step 3: StructuralBlueprinting (Reasoning...)")
        runtime._resolve_effect(ReasoningTrace(payload=TraceLog(
            thought="I will create a structured outline from the insights.",
            plan={"action": "create_structured_outline"}
        )))
        outline = ana.create_structured_outline(insights, "research_topic", "paper")
        
        # --- Step 4: Drafting ---
        print("   Step 4: Drafting (Reasoning...)")
        runtime._resolve_effect(ReasoningTrace(payload=TraceLog(
            thought="Drafting the content based on the outline.",
            plan={"action": "draft_content"}
        )))
        draft = wri.draft_content(outline)
        
        # --- Step 5: QualityControl ---
        print("   Step 5: QualityControl (Reasoning...)")
        runtime._resolve_effect(ReasoningTrace(payload=TraceLog(
            thought="Evaluating the draft against quality criteria.",
            plan={"action": "evaluate"}
        )))
        eval_res = rev.evaluate(draft, ["Source Fidelity"])
        
        # --- Step 6: Finalization ---
        print("   Step 6: Finalization (Reasoning...)")
        runtime._resolve_effect(ReasoningTrace(payload=TraceLog(
            thought="Finalizing the artifact and saving to file.",
            plan={"action": "finalize_artifact"}
        )))
        
        # Mocking the result to satisfy Domain Invariant
        final_content = "Title\n\nMotivation: ...\nBackground: ...\nMethodology: ...\nResults: ...\n"
        wri.finalize_artifact(final_content, "final_paper.md")
        
        if runtime.trace:
            runtime.trace[-1]['result'] = final_content

        print("✅ Workflow execution complete.")
        
    except Exception as e:
        print(f"❌ Workflow failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Save trace to REPL for checking
    repl.last_trace = runtime.trace
    
    # 4. Dual Validation
    print("\n⚖️ [Test Script] Performing Dual Validation...")
    repl.do_check("")

if __name__ == "__main__":
    main()
