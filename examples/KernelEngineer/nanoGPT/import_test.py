import os
import sys

# Attempt 1: Add the src folder to sys.path
tilegym_src = os.path.join(os.path.dirname(__file__), 'TileGym', 'src')
if os.path.exists(tilegym_src):
    sys.path.append(tilegym_src)
    print(f"Added to path: {tilegym_src}")

print("--- Import Test ---")
try:
    import tilegym
    print("SUCCESS: 'import tilegym' worked.")
except Exception as e:
    print(f"FAIL: 'import tilegym' failed. Error: {e}")

try:
    from tilegym.ops.cutile import attention
    print("SUCCESS: 'from tilegym.ops.cutile import attention' worked.")
except Exception as e:
    print(f"FAIL: could not import attention kernel. Error: {e}")

try:
    from tilegym.ops import fmha
    print("SUCCESS: 'from tilegym.ops import fmha' worked.")
except Exception as e:
    print(f"FAIL: could not import ops dispatcher. Error: {e}")

print("--- Directory Check ---")
ops_path = os.path.join(tilegym_src, 'tilegym', 'ops', 'cutile')
if os.path.exists(ops_path):
    print(f"Directory exists: {ops_path}")
    print(f"Contents: {os.listdir(ops_path)}")
else:
    print(f"Directory NOT FOUND: {ops_path}")
