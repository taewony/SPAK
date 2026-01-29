import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from kernel.dsl.parser import DSLParser
from kernel.vm import StepMachine

def main():
    parser = argparse.ArgumentParser(description="SPAK Step-wise Agent Kernel")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Command: run
    run_parser = subparsers.add_parser("run", help="Run a DSL task")
    run_parser.add_argument("file", help="Path to .dsl file")
    run_parser.add_argument("--backend", default="sim", choices=["sim", "ollama"], help="LLM Backend")
    run_parser.add_argument("--live", action="store_true", help="Enable Real Tool Execution (Subprocess)")

    # Command: step (with resume)
    step_parser = subparsers.add_parser("step", help="Step-wise execution control")
    step_parser.add_argument("file", nargs="?", help="Path to .dsl file (to locate context)")
    step_parser.add_argument("--resume", action="store_true", help="Resume from suspended state")
    
    args = parser.parse_args()

    if args.command == "run":
        if not os.path.exists(args.file):
            print(f"Error: File {args.file} not found.")
            return

        print(f"[*] Compiling {args.file}...")
        dsl_parser = DSLParser()
        try:
            spec = dsl_parser.parse_file(args.file)
            task = spec.task
        except Exception as e:
            print(f"Error Parsing DSL: {e}")
            return
        
        # New Run: Clear old context in target dir
        target_dir = os.path.dirname(os.path.abspath(args.file))
        ctx_path = os.path.join(target_dir, "context.json")
        res_path = os.path.join(target_dir, "response.txt")

        if os.path.exists(ctx_path):
            print(f"[!] Found existing context.json in {target_dir}. Overwriting for new run.")
            os.remove(ctx_path)
        if os.path.exists(res_path):
            os.remove(res_path)

        vm = StepMachine(task, dsl_path=args.file, backend=args.backend, live_mode=args.live)
        vm.run()

    elif args.command == "step":
        target_dir = os.getcwd()
        if args.file:
             target_dir = os.path.dirname(os.path.abspath(args.file))
        
        context_path = os.path.join(target_dir, "context.json")

        if args.resume:
            if not os.path.exists(context_path):
                print(f"Error: No context.json found in {target_dir}. Cannot resume.")
                return
            
            import json
            with open(context_path, "r", encoding="utf-8") as f:
                ctx = json.load(f)
                dsl_file = ctx.get("dsl_path")
            
            if not dsl_file or not os.path.exists(dsl_file):
                 print(f"Error: Could not find DSL file '{dsl_file}' from context.")
                 return

            print(f"[*] Resuming task from {dsl_file}...")
            
            dsl_parser = DSLParser()
            try:
                spec = dsl_parser.parse_file(dsl_file)
                task = spec.task
            except Exception as e:
                print(f"Error Parsing DSL: {e}")
                return
            
            # Note: We assume backend is same as before or default? 
            # Context doesn't store backend choice currently. Default to sim.
            vm = StepMachine(task, resume_state_path=context_path, dsl_path=dsl_file)
            vm.run()
        else:
            print("Usage: spak_v2.py step --resume")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
