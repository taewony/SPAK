import sys
import os

# Add the project root to sys.path to ensure 'kernel' package is resolvable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from kernel.spec_repl import SpecREPL

if __name__ == "__main__":
    SpecREPL().cmdloop()
