from kernel.compiler import Compiler
import os

def test_dsl_compilation():
    compiler = Compiler()
    
    test_files = [
        "design/examples/WebPageGen/04_date_app.dsl",
        "specs/research_agent.spec",
        "specs/matmul_kernel.spec"
    ]
    
    print("🔍 Testing DSL Compilation...")
    
    for file_path in test_files:
        if not os.path.exists(file_path):
            print(f"⚠️ File not found: {file_path}")
            continue
            
        print(f"\n📂 Compiling {file_path}...")
        try:
            spec = compiler.compile_file(file_path)
            print(f"✅ Success! Parsed System: {spec.name}")
            
            if spec.pages:
                print(f"   - Found {len(spec.pages)} pages")
            if spec.gpu_kernels:
                print(f"   - Found {len(spec.gpu_kernels)} GPU kernels")
            if spec.system_model:
                print(f"   - System Model: {spec.system_model.name}")
                
        except Exception as e:
            print(f"❌ Compilation Failed: {e}")
            # Raise to fail the test
            raise e

if __name__ == "__main__":
    test_dsl_compilation()
