import sys
import os
import json

# Ensure project root is in path
sys.path.append(os.getcwd())

from kernel.semantic_kernel import perform
import kernel.semantic_kernel as sk
from kernel.runtime import Runtime
from kernel.effects import ReadFile, FileRead, Generate
from kernel.handlers import (
    LiteLLMHandler, FileSystemHandler, MathHandler, 
    UserInteractionHandler, ReasoningHandler, SafeREPLHandler, Handler, Effect
)

# Components
from .librarian import Librarian
from .analyst import Analyst
from .writer import Writer
from .reviewer import Reviewer

# --- Custom Mock Handler for Chef Flow ---
class ChefMockHandler(Handler):
    def handle(self, effect: Effect) -> any:
        if isinstance(effect, Generate):
            prompt = effect.payload.messages[0]['content']
            
            # Analyst.extract_insights
            if "Extract key insights" in prompt:
                return json.dumps([
                    {
                        "claim": "SPAK Architecture uses Meta-IR and Domain-IR.",
                        "evidence": "Defined in AgentSpec.md",
                        "source": "docs/AgentSpec.md",
                        "confidence": 0.95
                    },
                    {
                        "claim": "The Librarian uses Hybrid Tooling (grep).",
                        "evidence": "Source code analysis",
                        "source": "src/librarian.py",
                        "confidence": 0.9
                    }
                ])
            
            # Analyst.create_structured_outline
            if "Structural Architect" in prompt:
                return """# Introduction
- Overview of SPAK
- Problem statement

# Proposed Method
- The Knowledge Chef Architecture
- Hybrid Tooling

# Conclusion
- Summary of benefits
"""

            # Writer.draft_content
            if "Technical Writer" in prompt:
                return """# Introduction
SPAK stands for Systemic Protocol for Agent Knowledge. It addresses the chaos of unstructured LLM agents.

# Proposed Method
We introduce the 'Knowledge Chef' architecture. It uses a Librarian to gather ingredients and an Analyst to cook them.

# Conclusion
This architecture proves that structure beats raw power.
"""

            # Reviewer.evaluate
            if "Strict Reviewer" in prompt:
                return json.dumps({
                    "status": "PASS",
                    "feedback": "Excellent structure and clarity.",
                    "score": 9.0
                })

            # Writer.refine_content
            if "Refine the following draft" in prompt:
                return "Refined Content: " + prompt[:50] + "..."

            return "Mock Response"
        raise NotImplementedError

def main():
    print("👨‍🍳 [Chef Mock] Initializing Kitchen (Runtime)...")
    
    # 1. Setup Runtime
    runtime = Runtime()
    # Use Mock Handler instead of LiteLLM
    runtime.register_handler(ChefMockHandler())
    runtime.register_handler(FileSystemHandler())
    runtime.register_handler(ReasoningHandler())
    sk._active_runtime = runtime

    # 2. Instantiate Staff
    lib = Librarian()
    analyst = Analyst()
    writer = Writer()
    reviewer = Reviewer()

    # 3. Configuration
    SOURCE_DIR = "docs"
    KEYWORDS = "Agent, SPAK"
    OUTLINE_TEMPLATE_PATH = "docs/paper_outline.md"
    OUTPUT_PATH = "output/SPAK_KISS_Paper_Mock.md"

    print(f"👨‍🍳 [Chef Mock] Order Received: Write a paper using template '{OUTLINE_TEMPLATE_PATH}'")

    # --- Step 1: Prep Ingredients (Librarian) ---
    print("\n📦 [1/5] Gathering Materials...")
    docs = lib.gather_materials(SOURCE_DIR, KEYWORDS)
    print(f"    -> Found {len(docs)} relevant documents.")
    # In mock mode, if no docs found, we might want to inject dummy docs to proceed
    if not docs:
        print("⚠️ No docs found in 'docs/', using dummy data for mock flow.")
        docs = ["Dummy content about SPAK Agent"]

    # --- Step 2: Cook (Analyst) ---
    print("\n🍳 [2/5] Extracting Insights...")
    try:
        insights = analyst.extract_insights(docs)
        print(f"    -> Extracted {len(insights)} structured insights.")
    except Exception as e:
        print(f"❌ Analysis Failed: {e}")
        return

    print("\n📐 [3/5] Blueprinting (Structural Architect)...")
    try:
        # Read the template file
        if os.path.exists(OUTLINE_TEMPLATE_PATH):
            template_content = perform(ReadFile(FileRead(path=OUTLINE_TEMPLATE_PATH)))
        else:
            template_content = "Dummy IMRaD Template"
    except Exception as e:
        print(f"❌ Could not read template: {e}")
        return

    structured_outline = analyst.create_structured_outline(
        insights, 
        intent="Write a comprehensive academic paper on SPAK Architecture", 
        structure_type=template_content
    )
    print("    -> Blueprint created.")

    # --- Step 3: Plate (Writer) ---
    print("\n✍️ [4/5] Drafting Content...")
    draft = writer.draft_content(structured_outline)
    print("    -> First draft generated.")
    print(f"    (Preview): {draft[:100]}...")

    # --- Step 4: Taste Test (Reviewer Loop) ---
    print("\n🧐 [5/5] Quality Control (Reviewer)...")
    max_retries = 2
    for i in range(max_retries):
        eval_result = reviewer.evaluate(draft, criteria=[
            "Follows KISS structure",
            "Technical depth"
        ])
        
        print(f"    [Round {i+1}] Score: {eval_result.score}/10 | Status: {eval_result.status}")
        print(f"    Feedback: {eval_result.feedback}")

        if eval_result.status == "PASS" or eval_result.score >= 8.5:
            print("    ✅ Quality Control Passed!")
            break
        
        if i < max_retries - 1:
            print("    🔧 Refining draft based on feedback...")
            draft = writer.refine_content(draft, eval_result.feedback)
    
    # --- Finalize ---
    print(f"\n🚀 Serving Dish to {OUTPUT_PATH}...")
    writer.finalize_artifact(draft, OUTPUT_PATH)
    print("👨‍🍳 [Chef Mock] Bon Appétit! (Process Complete)")

if __name__ == "__main__":
    main()
