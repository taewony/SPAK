import cmd
import os
import sys
from .compiler import AISpecCompiler
from .verifier import Verifier

class SpecREPL(cmd.Cmd):
    intro = 'Welcome to the Spec-Driven Build Agent Shell. Type help or ? to list commands.\n'
    prompt = '(kernel) '

    def __init__(self):
        super().__init__()
        self.compiler = AISpecCompiler()
        self.verifier = Verifier()
        self.current_spec = None
        self.spec_path = None

    def do_load(self, arg):
        """Load a spec file. Usage: load specs/SPEC.root.md"""
        if not arg:
            print("Please provide a file path.")
            return
        
        if not os.path.exists(arg):
            print(f"File not found: {arg}")
            return

        try:
            self.current_spec = self.compiler.compile_file(arg)
            self.spec_path = arg
            print(f"Successfully loaded System: '{self.current_spec.name}'")
            print(f"Components: {[c.name for c in self.current_spec.components]}")
        except Exception as e:
            print(f"Error parsing spec: {e}")

    def do_verify(self, arg):
        """Verify the implementation against the loaded spec. Usage: verify [src_dir]"""
        if not self.current_spec:
            print("No spec loaded. Use 'load <file>' first.")
            return

        src_dir = arg if arg else "src"
        if not os.path.exists(src_dir):
            print(f"Source directory '{src_dir}' not found.")
            return

        print(f"Verifying '{self.current_spec.name}' against '{src_dir}'...")
        self.verifier.verify_spec(self.current_spec, src_dir)
    
    def do_implement(self, arg):
        """Auto-implement missing components using the Builder (LLM)."""
        if not self.current_spec:
            print("No spec loaded.")
            return
        
        # This would call Builder logic. For now, just a placeholder message.
        print("[Builder] implementation logic is not yet fully connected to LLM.")
        print("[Builder] Please verify manually or run 'verify structure' to see gaps.")

    def do_show(self, arg):
        """Show details of the loaded spec."""
        if not self.current_spec:
            print("No spec loaded.")
            return
        
        print(f"System: {self.current_spec.name}")
        for comp in self.current_spec.components:
            print(f"\n  Component: {comp.name}")
            if comp.description:
                print(f"    Desc: {comp.description}")
            if comp.states:
                print(f"    State Models: {[s.name for s in comp.states]}")
            if comp.functions:
                print("    Functions:")
                for f in comp.functions:
                    params = ", ".join([f"{p.name}: {p.type.name}" for p in f.params])
                    print(f"      - {f.name}({params}) -> {f.return_type.name}")
            if comp.invariants:
                print("    Invariants:")
                for i in comp.invariants:
                    print(f"      * {i}")

    def do_exit(self, arg):
        """Exit the shell."""
        print("Goodbye.")
        return True

if __name__ == '__main__':
    SpecREPL().cmdloop()