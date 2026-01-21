import sys
import os
import json

# Ensure project root is in path
sys.path.append(os.getcwd())

from kernel.semantic_kernel import perform
import kernel.semantic_kernel as sk
from kernel.runtime import Runtime
from kernel.effects import ReadFile, FileRead
from kernel.handlers import (
    LiteLLMHandler, FileSystemHandler, MathHandler, 
    UserInteractionHandler, ReasoningHandler, SafeREPLHandler
)

# Components
from .librarian import Librarian
from .analyst import Analyst
from .writer import Writer
from .reviewer import Reviewer

from .chef_mock import ChefMockHandler

def run_chef(
    model_name: str = "ollama/qwen3:8b", 
    source_dir: str = "docs", 
    template_path: str = "docs/paper_outline.md", 
    output_path: str = "output/SPAK_KISS_Paper.md",
    use_mock: bool = False
):
    print("👨‍🍳 [Chef] Initializing Kitchen (Runtime)...")
    
    # 1. Setup Runtime
    runtime = Runtime()
    if use_mock:
        runtime.register_handler(ChefMockHandler())
    else:
        runtime.register_handler(LiteLLMHandler(default_model=model_name)) 
    
    runtime.register_handler(FileSystemHandler())
    runtime.register_handler(ReasoningHandler())
    sk._active_runtime = runtime

    # 2. Instantiate Staff
    lib = Librarian()
    analyst = Analyst()
    writer = Writer()
    reviewer = Reviewer()

    # Configuration
    KEYWORDS = "Agent, SPAK, Architecture, LLM"

    print(f"👨‍🍳 [Chef] Order Received: Write a paper using template '{template_path}'")
    print(f"    - Model: {model_name}")
    print(f"    - Source: {source_dir}")

    # --- Step 1: Prep Ingredients (Librarian) ---
    print("\n📦 [1/5] Gathering Materials...")
    docs = lib.gather_materials(source_dir, KEYWORDS)
    print(f"    -> Found {len(docs)} relevant documents.")
    if not docs:
        print("❌ No documents found. Aborting.")
        return

    # --- Step 2: Cook (Analyst) ---
    print("\n🍳 [2/5] Extracting Insights...")
    try:
        insights = analyst.extract_insights(docs)
        print(f"    -> Extracted {len(insights)} structured insights.")
    except Exception as e:
        print(f"❌ Analysis Failed: {e}")
        if "CUDA" in str(e) or "allocate" in str(e):
            print("💡 TIP: Your GPU might be out of memory.")
            print(f"   Try pulling a smaller model: 'ollama pull qwen2.5:1.5b'")
            print("   Then run with that model.")
        return

    print("\n📐 [3/5] Blueprinting (Structural Architect)...")
    try:
        # Read the template file
        if os.path.exists(template_path):
            template_content = perform(ReadFile(FileRead(path=template_path)))
        else:
            print(f"⚠️ Template '{template_path}' not found. Using default structure.")
            template_content = "Academic Paper Structure (IMRaD)"
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

    # --- Step 4: Taste Test (Reviewer Loop) ---
    print("\n🧐 [5/5] Quality Control (Reviewer)...")
    max_retries = 2
    for i in range(max_retries):
        eval_result = reviewer.evaluate(draft, criteria=[
            "Follows KISS (Korea Information Science Society) structure",
            "Technical depth and accuracy",
            "Clear distinction between Meta-IR and Domain-IR"
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
    print(f"\n🚀 Serving Dish to {output_path}...")
    writer.finalize_artifact(draft, output_path)
    print("👨‍🍳 [Chef] Bon Appétit! (Process Complete)")

def main():
    run_chef()

if __name__ == "__main__":
    main()
