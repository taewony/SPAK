import json
import os
import sys
from typing import Dict, Any, List

# Add project root to path
sys.path.append(os.getcwd())

from kernel.semantic_kernel import perform
from kernel.effects import Generate, LLMRequest
from kernel.compiler import Compiler

# Sub-Components
from src.librarian import Librarian
from src.analyst import Analyst
from src.writer import Writer
from src.reviewer import Reviewer

class ChiefEditor:
    """
    Level 4 Agent: ChiefEditor
    Dynamically plans and orchestrates the creation of a research report.
    Uses the System Model from 'specs/knowledge_chef.spec.md' to guide reasoning.
    """
    
    def __init__(self):
        # Instantiate Sub-Agents (The Staff)
        self.librarian = Librarian()
        self.analyst = Analyst()
        self.writer = Writer()
        self.reviewer = Reviewer()
        
        # Blackboard State
        self.state = {
            "topic": None,
            "docs": [],
            "insights": [],
            "outline": None,
            "draft": None,
            "evaluation": None,
            "output_path": None,
            "history": [] # Track actions taken
        }

        # Load System Model
        self.system_model_text = self._load_system_model()

    def _load_system_model(self) -> str:
        try:
            compiler = Compiler()
            spec_path = "specs/knowledge_chef.spec.md"
            if os.path.exists(spec_path):
                spec = compiler.compile_file(spec_path)
                if spec.system_model:
                    lines = []
                    for stmt in spec.system_model.statements:
                        lines.append(f"[{stmt.type.upper()}] {stmt.content}")
                    return "\n".join(lines)
            return "No System Model found."
        except Exception as e:
            print(f"⚠️ Failed to load System Model: {e}")
            return ""

    def synthesize_report(self, topic: str, source_dir: str = "docs", output_path: str = "output/report.md"):
        self.state["topic"] = topic
        self.state["output_path"] = output_path
        
        print(f"🎓 [ChiefEditor] Starting synthesis on '{topic}'")
        print(f"🧠 [System Model Loaded]\n{self.system_model_text}\n")

        max_steps = 10
        step_count = 0
        
        while step_count < max_steps:
            step_count += 1
            print(f"\n🔄 [ChiefEditor] Planning Step {step_count}...")
            
            # 1. Plan
            plan = self._plan_next_step(source_dir)
            
            # 2. Execute
            if not plan:
                print("❌ Planner returned no plan. Aborting.")
                break
                
            print(f"    👉 Thought: {plan.get('thought')}")
            print(f"    👉 Action: {plan.get('component')}.{plan.get('function')}")
            
            if plan.get('function') == "DONE":
                print("✅ [ChiefEditor] Workflow Complete.")
                break
                
            self._execute_step(plan)
            
        if step_count >= max_steps:
            print("⚠️ [ChiefEditor] Max steps reached. Terminating.")

    def _plan_next_step(self, source_dir: str) -> Dict[str, Any]:
        # Summarize State for LLM
        state_summary = {
            "topic": self.state["topic"],
            "has_docs": len(self.state["docs"]) > 0,
            "docs_count": len(self.state["docs"]),
            "has_insights": len(self.state["insights"]) > 0,
            "has_outline": self.state["outline"] is not None,
            "has_draft": self.state["draft"] is not None,
            "last_evaluation": self.state["evaluation"].__dict__ if self.state["evaluation"] else "None"
        }
        
        prompt = f"""You are the Chief Editor. Manage the workflow to produce a high-quality report.

GOAL: Synthesize a report on '{self.state['topic']}'.

CURRENT STATE:
{json.dumps(state_summary, indent=2)}

AVAILABLE TOOLS:
1. Librarian.gather_materials(source_path=\"{source_dir}\", topic_keywords=\"...\")
2. Analyst.extract_insights(documents=[...]) 
   (Pre-condition: Must have docs)
3. Analyst.create_structured_outline(insights=[...], intent=\"...\", structure_type=\"IMRaD\")
   (Pre-condition: Must have insights)
4. Writer.draft_content(outline=\"...\")
   (Pre-condition: Must have outline)
5. Reviewer.evaluate(draft=\"...\", criteria=[\"Technical Accuracy\", \"Citation Compliance\"])
   (Pre-condition: Must have draft)
6. Writer.refine_content(draft=\"...\", feedback=\" கிலோ\" ) # Fixed: Changed feedback arg to be more descriptive
   (Pre-condition: Must have draft and negative evaluation)
7. Writer.finalize_artifact(content=\"...\", path=\"{self.state['output_path']}\")
   (Pre-condition: Must have draft and positive evaluation (PASS or Score > 8.0))
8. DONE (Call this only after finalizing the artifact)

INSTRUCTIONS:
- Analyze the Current State.
- Choose the logically next step.
- Return strictly JSON:
{{
  "thought": "Reasoning for this step...",
  "component": "ComponentName", 
  "function": "function_name",
  "args": {{ "arg_name": "value" }}
}}
- If the function is DONE, set component/function to "DONE".
"""
        # Inject System Model via LLMRequest
        response = perform(Generate(LLMRequest(
            messages=[{"role": "user", "content": prompt}],
            system_model=self.system_model_text
        )))
        
        try:
            # Clean JSON
            clean_json = response.strip()
            if "```json" in clean_json:
                clean_json = clean_json.split("```json")[1].split("```")[0]
            elif "```" in clean_json:
                clean_json = clean_json.split("```")[1].split("```")[0]
            
            return json.loads(clean_json.strip())
        except Exception as e:
            print(f"❌ Planning Error: {e}")
            print(f"Raw Response: {response}")
            return None

    def _execute_step(self, plan: Dict[str, Any]):
        comp_name = plan.get('component')
        func_name = plan.get('function')
        args = plan.get('args', {})
        
        # 1. Resolve Component
        component = None
        if comp_name == "Librarian": component = self.librarian
        elif comp_name == "Analyst": component = self.analyst
        elif comp_name == "Writer": component = self.writer
        elif comp_name == "Reviewer": component = self.reviewer
        
        if not component:
            print(f"❌ Unknown component: {comp_name}")
            return

        # 2. Resolve Arguments from Blackboard State
        # The planner might send placeholders or literal values. We need to inject large objects (docs, insights) from state.
        
        # Auto-inject state variables if args implies them
        if func_name == "extract_insights" and "documents" not in args:
            args["documents"] = self.state["docs"]
        elif func_name == "extract_insights" and args.get("documents") == "[...]": # Handle LLM placeholder
             args["documents"] = self.state["docs"]

        if func_name == "create_structured_outline" and "insights" not in args:
             args["insights"] = self.state["insights"]
        elif func_name == "create_structured_outline" and args.get("insights") == "[...]":
             args["insights"] = self.state["insights"]

        if func_name == "draft_content" and "outline" not in args:
             args["outline"] = self.state["outline"]
        elif func_name == "draft_content" and args.get("outline") == "...":
             args["outline"] = self.state["outline"]

        if func_name == "evaluate" and "draft" not in args:
             args["draft"] = self.state["draft"]
        elif func_name == "evaluate" and args.get("draft") == "...":
             args["draft"] = self.state["draft"]
        
        if func_name == "refine_content":
            if "draft" not in args or args.get("draft") == "...":
                args["draft"] = self.state["draft"]
            if "feedback" not in args or args.get("feedback") == "...":
                args["feedback"] = self.state["evaluation"].feedback if self.state["evaluation"] else ""

        if func_name == "finalize_artifact":
             if "content" not in args or args.get("content") == "...":
                 args["content"] = self.state["draft"]

        # 3. Call Method
        try:
            method = getattr(component, func_name)
            print(f"    ▶️ Executing {comp_name}.{func_name}...")
            result = method(**args)
            
            # 4. Update Blackboard
            if func_name == "gather_materials":
                self.state["docs"] = result
                print(f"    ✅ Updated State: docs ({len(result)} items)")
            elif func_name == "extract_insights":
                self.state["insights"] = result
                print(f"    ✅ Updated State: insights ({len(result)} items)")
            elif func_name == "create_structured_outline":
                self.state["outline"] = result
                print(f"    ✅ Updated State: outline set")
            elif func_name == "draft_content" or func_name == "refine_content":
                self.state["draft"] = result
                print(f"    ✅ Updated State: draft updated")
            elif func_name == "evaluate":
                self.state["evaluation"] = result
                print(f"    ✅ Updated State: evaluation ({result.status}, {result.score})")
            
        except Exception as e:
            print(f"❌ Execution Error: {e}")
