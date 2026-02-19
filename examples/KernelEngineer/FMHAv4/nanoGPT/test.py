import os
import sys

tilegym_src = os.path.join(os.path.dirname(__file__), 'TileGym', 'src', 'tilegym', 'ops', 'cutile')
if os.path.exists(tilegym_src):
    sys.path.append(tilegym_src)
    print(f"[INFO] Added {tilegym_src} to sys.path for TileGym ops.")


from tilegym.ops import fmha, layer_norm_legacy, matmul
print("[INFO] Successfully imported TileGym ops.")  
    
print("end")