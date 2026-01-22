import cmd
import os
import sys
import code
import importlib.util
from .compiler import Compiler
from .verifier import Verifier
from .builder import Builder

class SpecREPL(cmd.Cmd):
    intro = 'Welcome to the Spec-Driven Build Agent Shell. Type help or ? to list commands.\n'
    prompt = '(kernel) '

    def __init__(self):
        super().__init__()
        self.compiler = Compiler()
        self.verifier = Verifier()
        self.builder = Builder()
        self.current_specs = {}  # {name: spec}
        self.current_spec = None # active spec
        self.last_trace = []     # Store the execution trace from last run
        
        # Default Config
        self.config = {
            "model": "ollama/qwen3:8b",
            "language": "English",
            "goal": "becomming a CEO of AI-Powered Content Creation one-person company",
            "persona": "financial consultant",
            "autostart": True,
            "source": "docs",
            "target": "papers"
        }
        # Apply model config to builder
        self.builder.model_name = self.config["model"]
        
        # Show Config on Startup
        print("\n🔧 Current Configuration:")
        for k, v in self.config.items():
            print(f"   {k}: {v}")
        print("")

    def emptyline(self):
        pass

    def do_config(self, arg):
        """Manage configuration. Usage: config [key] [value]"""
        args = arg.split(' ', 1)
        if not arg:
            # Show all
            print("\n🔧 Current Configuration:")
            for k, v in self.config.items():
                print(f"   {k}: {v}")
            return

        key = args[0]
        if len(args) == 1:
            # Show specific
            if key in self.config:
                print(f"{key}: {self.config[key]}")
            else:
                print(f"Key '{key}' not found.")
        else:
            # Set value
            value = args[1]
            self.config[key] = value
            print(f"✅ Set '{key}' to '{value}'")
            
            # Side effects
            if key == "model":
                self.builder.model_name = value
                print(f"   (Updated Builder model to {value})")

    def do_trace(self, arg):
        """Display the execution trace. Usage: trace (thoughts only) OR trace all (full log)"""
        if not self.last_trace:
            print("No trace available. Run an agent first.")
            return
        
        show_all = arg.strip() == "all"
        
        from rich.console import Console
        from rich.table import Table
        from rich.syntax import Syntax
        import json

        console = Console()
        title = "Execution Trace (Full System Effects)" if show_all else "Execution Trace (reasoning & Plan Only)"
        table = Table(title=title)
        table.add_column("Step", justify="right", style="cyan", no_wrap=True)
        table.add_column("Effect", style="magenta")
        table.add_column("Payload", style="green")
        table.add_column("Result", style="blue")

        for i, entry in enumerate(self.last_trace):
            name = entry['name']
            payload = entry['payload']
            result = entry.get('result', 'N/A')
            
            # Filter logic: If not 'all', skip anything that isn't reasoning
            if not show_all and name != "ReasoningTrace":
                continue

            # Special highlighting for ReasoningTrace
            if name == "ReasoningTrace":
                thought = getattr(payload, 'thought', 'N/A')
                plan = getattr(payload, 'plan', {})
                raw = getattr(payload, 'raw_response', '')
                
                payload_str = f"[bold yellow]Thought:[/bold yellow] {thought}\n[bold cyan]Plan:[/bold cyan] {plan}"
                if show_all and raw:
                     payload_str += f"\n[dim]Raw LLM:[/dim] {raw[:100]}..."
            elif name == "Generate" and show_all:
                # Provide better formatting for Generate (LLM Request)
                msgs = getattr(payload, 'messages', [])
                payload_str = f"Generate({len(msgs)} messages)"
            else:
                payload_str = str(payload)
            
            # Format result string safely
            result_str = str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
            
            table.add_row(str(i+1), name, payload_str, result_str)
        
        console.print(table)

    def do_load(self, arg):
        """Load spec file(s). Usage: load specs/SPEC.root.md OR load specs"""
        if not arg:
            print("Please provide a file or directory path.")
            return
        
        if not os.path.exists(arg):
            print(f"Path not found: {arg}")
            return

        if os.path.isdir(arg):
            count = 0
            for root, _, files in os.walk(arg):
                for file in files:
                    if file.endswith(".spec.md"):
                        path = os.path.join(root, file)
                        self._load_single_file(path)
                        count += 1
            if count == 0:
                print(f"No SPEC files found in {arg}")
            else:
                print(f"Loaded {count} specs from {arg}")
                if self.current_specs:
                    # Prefer SPAK_Kernel if loaded, else last one
                    if "SPAK_Kernel" in self.current_specs:
                        self.current_spec = self.current_specs["SPAK_Kernel"]
                    else:
                        self.current_spec = list(self.current_specs.values())[-1]
                    print(f"Active System: '{self.current_spec.name}'")
        else:
            self._load_single_file(arg)

    def _load_single_file(self, path):
        try:
            spec = self.compiler.compile_file(path)
            self.current_specs[spec.name] = spec
            self.current_spec = spec
            print(f"Successfully loaded System: '{spec.name}' from {path}")
        except Exception as e:
            print(f"Error parsing {path}: {e}")

    def do_list(self, arg):
        """List loaded specs."""
        if not self.current_specs:
            print("No specs loaded.")
            return
        print("Loaded Systems:")
        for name in self.current_specs:
            prefix = "*" if self.current_spec and self.current_spec.name == name else " "
            print(f"{prefix} {name}")

    def do_use(self, arg):
        """Set active spec. Usage: use SystemName"""
        if arg in self.current_specs:
            self.current_spec = self.current_specs[arg]
            print(f"✅ Active System set to: '{arg}'")
            
            src_exists = False
            test_exists = False
            
            for comp in self.current_spec.components:
                if os.path.exists(os.path.join("src", f"{comp.name.lower()}.py")):
                    src_exists = True
                    break
            
            test_file = os.path.join("tests", f"tests.{self.current_spec.name.lower().replace('system','')}.yaml")
            if os.path.exists(test_file):
                test_exists = True

            print("\n👉 Next Possible Actions:")
            if not src_exists:
                print(f"   - Run 'build' to generate implementation and tests.")
            elif not test_exists:
                print(f"   - Run 'build' to generate missing tests.")
            else:
                print(f"   - Run 'verify' to check correctness.")
                print(f"   - Run 'run [Component]' to interact with the built agent.")

        else:
            print(f"System '{arg}' not found. Loaded: {list(self.current_specs.keys())}")

    def do_verify(self, arg):
        """Verify the implementation against the loaded spec. Usage: verify [src_dir]"""
        if not self.current_spec:
            print("No active spec.")
            return

        src_dir = arg if arg else "src"
        if not os.path.exists(src_dir):
            os.makedirs(src_dir, exist_ok=True)

        print(f"Verifying '{self.current_spec.name}' against '{src_dir}'...")
        self.verifier.verify_spec(self.current_spec, src_dir)
    
    def do_build(self, arg):
        """Auto-implement missing components AND generate tests using TDD flow. Usage: build [src_dir]"""
        if not self.current_spec:
            print("No active spec.")
            return
        
        src_dir = arg if arg else "src"
        test_dir = "tests"
        if not os.path.exists(src_dir): os.makedirs(src_dir, exist_ok=True)
        if not os.path.exists(test_dir): os.makedirs(test_dir, exist_ok=True)

        print(f"🚀 [Kernel] Starting Spec-Driven TDD Build Process...")

        print(f"📋 [Kernel] Phase 1: Test Generation")
        test_contents = {}

        for comp in self.current_spec.components:
            test_file = os.path.join(test_dir, f"tests.{comp.name.lower()}.yaml")
            
            if not os.path.exists(test_file):
                print(f"  generating tests for {comp.name}...")
                yaml_content = self.builder.generate_tests(comp, self.current_spec.name)
                with open(test_file, "w", encoding="utf-8") as f:
                    f.write(yaml_content)
                print(f"  ✅ Created {test_file}")
                test_contents[comp.name] = yaml_content
            else:
                print(f"  ℹ️  Using existing tests for {comp.name}")
                with open(test_file, "r", encoding="utf-8") as f:
                    test_contents[comp.name] = f.read()

        print(f"\n🏗️ [Kernel] Phase 2: Implementation (Test-Guided)")
        
        errors = self.verifier.static.verify(self.current_spec, src_dir)
        missing_components = []
        for err in errors:
            if "Missing implementation for Component" in err:
                comp_name = err.split("'")[1]
                comp_ast = next((c for c in self.current_spec.components if c.name == comp_name), None)
                if comp_ast:
                    missing_components.append(comp_ast)
        
        if not missing_components:
            print("✨ All components are already implemented. (Run 'repair' if logic is broken)")
            return

        for comp in missing_components:
            test_context = ""
            if comp.name in test_contents:
                test_context = f"\nCRITICAL: The implementation MUST pass the following tests:\n\n{test_contents[comp.name]}"
            
            code = self.builder.implement_component(comp, test_context)
            
            file_name = f"{comp.name.lower()}.py"
            file_path = os.path.join(src_dir, file_name)
            
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code)
            
            print(f"✅ Synthesized {file_name} (Aligned with tests)")

        print("\n🏁 [Kernel] TDD Build complete. Run 'verify' to confirm.")

    def do_repair(self, arg):
        """Attempt to repair implementation OR tests based on verification errors. Usage: repair [src_dir]"""
        if not self.current_spec:
            print("No active spec.")
            return

        src_dir = arg if arg else "src"
        test_dir = "tests"
        
        print(f"🔧 [Kernel] Running diagnosis on '{self.current_spec.name}'...")
        
        dynamic_errors = []
        # Iterate over components to find errors
        for comp in self.current_spec.components:
            test_file = os.path.join(test_dir, f"tests.{comp.name.lower()}.yaml")
            if os.path.exists(test_file):
                errs = self.verifier.verify_behavior(test_file, src_dir)
                for e in errs:
                    # Tag error with test file for context
                    dynamic_errors.append(f"[{test_file}] {e}")
        
        if not dynamic_errors:
            print("✨ [Kernel] No behavioral errors found. Checking structure...")
            static_errors = self.verifier.verify_structure(self.current_spec, src_dir)
            if not static_errors:
                print("✨ [Kernel] System is healthy. Nothing to repair.")
                return
            print(f"⚠️ [Kernel] Found {len(static_errors)} structural errors. (Repairing structural mismatch is pending feature)")
            return

        print(f"🚨 [Kernel] Found {len(dynamic_errors)} behavioral/runtime errors. Analyzing root cause...")
        
        error_log = "\n".join(dynamic_errors)
        
        test_failure_keywords = [
            "unexpected keyword argument",
            "missing 1 required positional argument",
            "takes 0 positional arguments but",
            "module 'typing' has no attribute"
        ]
        
        is_test_issue = any(k in error_log for k in test_failure_keywords)
        
        if is_test_issue:
            print("🧐 [Kernel] Diagnosis: The TESTS seem to be calling functions incorrectly.")
            
            # Identify which test file is broken
            # For simplicity, we repair all test files involved in errors
            involved_test_files = set()
            for err in dynamic_errors:
                if "[" in err:
                    path = err.split("]")[0].strip("[")
                    involved_test_files.add(path)
            
            for test_file in involved_test_files:
                print(f"🚑 [Kernel] Repairing Test File '{test_file}'...")
                with open(test_file, 'r', encoding='utf-8') as f:
                    broken_yaml = f.read()
                
                fixed_yaml = self.builder.fix_tests(broken_yaml, error_log)
                
                with open(test_file, 'w', encoding='utf-8') as f:
                    f.write(fixed_yaml)
                print(f"✅ [Kernel] Applied fix to tests.")
            
        else:
            print("🧐 [Kernel] Diagnosis: The IMPLEMENTATION seems to have logic errors.")
            
            # Similar logic for implementation files
            for comp in self.current_spec.components:
                test_file = os.path.join(test_dir, f"tests.{comp.name.lower()}.yaml")
                test_context = ""
                if os.path.exists(test_file):
                    with open(test_file, 'r', encoding='utf-8') as f:
                        test_context = f"\n\nRELATED TEST FILE ({test_file}):\n{f.read()}"

                file_name = f"{comp.name.lower()}.py"
                file_path = os.path.join(src_dir, file_name)
                
                if os.path.exists(file_path):
                    # Only repair if errors seem related to this component? 
                    # For now, simplistic approach: repair all if generic errors.
                    print(f"🚑 [Kernel] Repairing Implementation '{file_path}'...")
                    with open(file_path, 'r', encoding='utf-8') as f:
                        broken_code = f.read()
                    
                    full_context = error_log + test_context
                    fixed_code = self.builder.fix_implementation(broken_code, full_context)
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(fixed_code)
                    print(f"✅ [Kernel] Applied fix to implementation.")

        print("\n🏁 [Kernel] Repair sequence complete. Run 'verify' to check if it worked.")

    def do_history(self, arg):
        """Show LLM conversation history. Usage: history [last_n]"""
        history = self.builder.get_history()
        if not history:
            print("No LLM interactions yet.")
            return
        
        limit = len(history)
        if arg.isdigit():
            limit = int(arg)
        
        print(f"\n📜 Showing last {limit} interactions:\n")
        for i, item in enumerate(history[-limit:]):
            print(f"--- [{i+1}] Type: {item['type']} ---")
            if 'component' in item:
                print(f"Component: {item['component']}")
            print(f"[PROMPT]:\n{item['prompt'][:200]}... (truncated)\n")
            print(f"[RESPONSE]:\n{item['response'][:200]}... (truncated)\n")

    def do_run(self, arg):
        """Run a component. Usage: run [SystemName] [ComponentName] [--mock] [args...]"""
        from . import semantic_kernel
        from .handlers import LiteLLMHandler, SafeREPLHandler, FileSystemHandler, MockLLMHandler, MathHandler, UserInteractionHandler, ReasoningHandler
        from .runtime import Runtime

        args = arg.split()

        # 1. Check if the first argument is a loaded System (Context Switch)
        if args and args[0] in self.current_specs:
            sys_name = args[0]
            self.current_spec = self.current_specs[sys_name]
            print(f"✅ Active System set to: '{sys_name}'")
            args.pop(0) # Consume the System Name
        
        if not self.current_spec:
            print("No active spec. Use 'load <path>' first.")
            return

        # 2. Determine Component Name
        # If no args left (or only flags?), default to first component
        # We need to separate flags from positional args potentially, but simple logic:
        # If args[0] starts with --, it's a flag, so default component.
        # Else, args[0] is component name.
        
        use_mock = "--mock" in args
        if use_mock:
            args.remove("--mock")
        
        if not args:
            comp_name = self.current_spec.components[0].name
            constructor_args = []
        else:
            comp_name = args[0]
            constructor_args = args[1:]
        
        # Auto-inject goal for Coach if not provided
        if comp_name == "Coach" and not constructor_args:
            if "goal" in self.config:
                constructor_args = [self.config["goal"]]
        
        # Auto-inject persona for PersonaChatBot if not provided
        if comp_name == "PersonaChatBot" and not constructor_args:
            if "persona" in self.config:
                constructor_args = [self.config["persona"]]

        src_dir = "src"
        module_path = os.path.join(src_dir, f"{comp_name.lower()}.py")
        
        if not os.path.exists(module_path):
            print(f"Implementation not found at {module_path}. Please build it first.")
            return

        print(f"🚀 Initializing {comp_name} Runtime{' (MOCK MODE)' if use_mock else ''}...")
        
        try:
            import inspect
            # Setup Kernel Runtime with default handlers
            runtime = Runtime()
            if use_mock:
                runtime.register_handler(MockLLMHandler())
            else:
                runtime.register_handler(LiteLLMHandler(default_model=self.builder.model_name))
            
            runtime.register_handler(SafeREPLHandler())
            runtime.register_handler(FileSystemHandler())
            runtime.register_handler(MathHandler())
            runtime.register_handler(UserInteractionHandler())
            runtime.register_handler(ReasoningHandler())
            
            # Set global runtime context so perform() works
            semantic_kernel._active_runtime = runtime

            spec = importlib.util.spec_from_file_location(comp_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            cls = getattr(module, comp_name)
            
            # Try instantiation
            try:
                instance = cls(*constructor_args)
            except TypeError as te:
                if "missing" in str(te) and "__init__" in str(te):
                    print(f"⚠️  Instantiation failed: {te}")
                    print(f"💡 Usage: run {comp_name} [arg1] [arg2]...")
                    semantic_kernel._active_runtime = None
                    return
                raise te
            
            print(f"✅ {comp_name} instantiated as 'app'.")
            
            # Inspect available methods with signatures
            methods = []
            for name, member in inspect.getmembers(instance, predicate=inspect.ismethod):
                if not name.startswith('_'):
                    sig = inspect.signature(member)
                    methods.append(f"{name}{sig}")
            
            if methods:
                print(f"💡 Available methods:")
                for m in methods:
                    print(f"   - app.{m}")
                
                # Intelligent example
                if any('calculate' in m for m in methods):
                    print(f"💡 Example: app.calculate(10, 5, 'add')")
                elif any('reply' in m for m in methods):
                    print(f"💡 Example: app.reply('Hello')")
                elif any('chat' in m for m in methods):
                    print(f"💡 Example: app.chat('Hello')")
            else:
                print(f"💡 Type python commands using 'app'.")

            print(f"💡 Type 'exit()' to return to kernel.")
            
            # Custom exit to return to kernel instead of killing process
            def back_to_kernel():
                print("Returning to SPAK Kernel...")
                return

            vars = globals().copy()
            vars.update(locals())
            vars['app'] = instance
            vars['runtime'] = runtime
            vars['exit'] = back_to_kernel
            vars['quit'] = back_to_kernel
            
            # Autostart Logic
            if self.config.get("autostart", False):
                if hasattr(instance, "start") and callable(instance.start):
                    print(f"▶️ Auto-starting {comp_name}...")
                    try:
                        # Call start()
                        result = instance.start()
                        print(f"Result: {result}")
                    except Exception as e:
                        print(f"Error during autostart: {e}")
            
            # Start interaction
            code.interact(local=vars, exitmsg="")
            
            # Save trace for do_trace command
            self.last_trace = runtime.trace
            
            # Cleanup
            semantic_kernel._active_runtime = None
            
        except Exception as e:
            print(f"Error running component: {e}")
            semantic_kernel._active_runtime = None

    def do_show(self, arg):
        """Show details of the active spec."""
        if not self.current_spec:
            print("No active spec.")
            return
        
        print(f"System: {self.current_spec.name}")
        for comp in self.current_spec.components:
            print(f"\n  Component: {comp.name}")
            if comp.description:
                print(f"    Desc: {comp.description}")
            # Show effects if available
            # Note: Need to update verifier/compiler AST to expose effects in component if attached
            # Currently effects are system level in AST.
            
            if comp.functions:
                print("    Functions:")
                for f in comp.functions:
                    params = ", ".join([f"{p.name}: {p.type.name}" for p in f.params])
                    print(f"      - {f.name}({params}) -> {f.return_type.name}")
                    if f.body:
                        print(f"        Body: {f.body}")

    def do_consistency(self, arg):
        """Verify the last run's trace against a PlanIR. Usage: consistency plans/my.plan.yaml"""
        if not self.last_trace:
            print("No trace available. Run an agent first (e.g., 'run ResearchAgent').")
            return
        
        if not arg or not os.path.exists(arg):
            print(f"Plan file not found: {arg}")
            return

        import yaml
        from .consistency import PlanIR, StepExpectation, ConsistencyVerifier

        try:
            with open(arg, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            steps = []
            for s in data.get('steps', []):
                steps.append(StepExpectation(
                    phase=s['phase'],
                    must_use_action=s['must_use_action'],
                    required_thought_keywords=s['required_thought_keywords']
                ))
            
            plan = PlanIR(name=data.get('name', 'Unnamed Plan'), steps=steps)
            
            verifier = ConsistencyVerifier()
            result = verifier.verify(self.last_trace, plan)
            
            print(f"\n📊 [Consistency Result] Score: {result['score'] * 100:.1f}%")
            if result['passed']:
                print("✅ PASSED: Execution Semantic Intent matches Plan.")
            else:
                print("❌ FAILED: Semantic Drift detected.")
            
            print("\nDetails:")
            for log in result['details']:
                print(log)

        except Exception as e:
            print(f"Error executing consistency check: {e}")

    def do_chef(self, arg):
        """Execute the Knowledge Chef workflow. Usage: chef [mock]"""
        try:
            from src.chef import run_chef
        except ImportError:
            print("❌ Could not import 'chef.py'. Ensure it exists in the 'src/' directory.")
            return

        is_mock = arg.strip().lower() == "mock"

        print(f"👨‍🍳 Starting Chef Workflow with config:")
        print(f"   Model: {self.config.get('model') if not is_mock else 'MOCK'}")
        print(f"   Source: {self.config.get('source')}")
        print(f"   Target: {self.config.get('target')}")

        target = self.config.get('target', 'output')
        if target.endswith('.md'):
            output_path = target
        else:
            if not os.path.exists(target):
                os.makedirs(target, exist_ok=True)
            output_path = os.path.join(target, "SPAK_Paper.md")

        try:
            run_chef(
                model_name=self.config.get('model'),
                source_dir=self.config.get('source'),
                template_path="docs/paper_outline.md",
                output_path=output_path,
                use_mock=is_mock
            )
        except Exception as e:
            print(f"❌ Chef execution failed: {e}")

    def do_check(self, arg):
        """Perform Dual Validation (Operational Consistency + Domain Invariants) on the last trace.
        Usage: check [plan_file]
        If plan_file is omitted, it attempts to derive the plan from the Active Spec's Workflow.
        """
        if not self.last_trace:
            print("No trace available. Run an agent first.")
            return

        if not self.current_spec:
            print("No active spec to check invariants against.")
            return

        print("\n🔎 Starting Dual Validation Process...\n")
        
        # --- 1. Operational Consistency (Trace <-> Plan) ---
        op_score = 0.0
        op_passed = False
        plan_file = arg.strip()
        plan_object = None

        from .consistency import PlanIR, StepExpectation, ConsistencyVerifier
        
        if plan_file and os.path.exists(plan_file):
            print(f"🔹 [Operational Consistency] Checking against plan file: {plan_file}")
            import yaml
            try:
                with open(plan_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                
                steps = []
                for s in data.get('steps', []):
                    steps.append(StepExpectation(
                        phase=s['phase'],
                        must_use_action=s['must_use_action'],
                        required_thought_keywords=s['required_thought_keywords']
                    ))
                plan_object = PlanIR(name=data.get('name', 'Unnamed Plan'), steps=steps)

            except Exception as e:
                print(f"   ⚠️ Error loading plan file: {e}")

        elif self.current_spec.workflows:
            # AUTO-DERIVE PLAN FROM SPEC
            wf = self.current_spec.workflows[0] # Take the first workflow as default
            print(f"🔹 [Operational Consistency] Deriving plan from Spec Workflow: '{wf.name}'")
            
            steps = []
            import re
            
            # Simple heuristic: Try to find 'perform Component.Function' in step logic
            for step_str in wf.steps:
                # Step string format from compiler: "Step {Name}: {Body}"
                # e.g. "Step SmartCollection: docs = perform Librarian.gather_materials(source_dir, intent)"
                
                parts = step_str.split(":", 1)
                if len(parts) < 2: continue
                
                step_name = parts[0].replace("Step", "").strip()
                step_body = parts[1].strip()
                
                # Extract Action
                action_match = re.search(r"perform\s+([a-zA-Z0-9_]+\.[a-zA-Z0-9_]+)", step_body)
                action = action_match.group(1).split('.')[-1] if action_match else "unknown_action"
                
                # Extract Keywords (Heuristic: Component Name + Action Name)
                # Ideally we read 'expect_thought' comments, but compiler strips them.
                # So we use the component/function names as keywords.
                keywords = [action.split('_')[-1], action.split('_')[0]] # e.g. gather_materials -> [materials, gather]
                
                steps.append(StepExpectation(
                    phase=step_name,
                    must_use_action=action,
                    required_thought_keywords=keywords
                ))
            
            plan_object = PlanIR(name=wf.name, steps=steps)
            
        if plan_object:
            try:
                verifier = ConsistencyVerifier()
                result = verifier.verify(self.last_trace, plan_object)
                
                op_score = result['score']
                op_passed = result['passed']
                print(f"   Score: {op_score * 100:.1f}%")
                if not op_passed:
                    print("   ❌ Failed Operational Consistency")
                    for d in result['details']: print(f"      {d}")
                else:
                    print("   ✅ Passed Operational Consistency")
            except Exception as e:
                 print(f"   ⚠️ Error running consistency verification: {e}")
        else:
             print("🔹 [Operational Consistency] Skipped (No plan file and no Workflows in Spec).")


        # --- 2. Domain Invariants (Trace <-> Spec) ---
        print(f"\n🔹 [Domain Invariants] Checking against Spec: {self.current_spec.name}")
        from .invariants import DomainInvariantChecker
        
        inv_checker = DomainInvariantChecker()
        inv_results = inv_checker.check_invariants(self.last_trace, self.current_spec)
        
        inv_passed_count = 0
        for res in inv_results:
            icon = "✅" if res.status == "PASS" else "❌" if res.status == "FAIL" else "❓"
            print(f"   {icon} [{res.status}] {res.name}")
            print(f"      {res.details}")
            if res.status == "PASS": inv_passed_count += 1
        
        # --- Summary ---
        print("\n" + "="*40)
        print("📝 Dual Validation Report")
        print("="*40)
        print(f"1. Operational Consistency: {'✅ PASS' if op_passed else '❌ FAIL' if plan_object else '⚪ SKIP'}")
        print(f"2. Domain Invariants:       {inv_passed_count}/{len(inv_results)} passed (verified)")
        
        overall = "PASS" if (op_passed or not plan_object) and all(r.status != 'FAIL' for r in inv_results) else "FAIL"
        print(f"\nOverall Status: {overall}")
        print("="*40 + "\n")

    def do_exit(self, arg):
        """Exit the shell."""
        print("Goodbye.")
        return True

if __name__ == '__main__':
    SpecREPL().cmdloop()