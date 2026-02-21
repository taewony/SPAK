import os
import pickle
import torch
import torch.nn.functional as F
from model_loop import LoopGPT
from model import GPTConfig

def analyze_thinking_pure_pytorch():
    # 1. Path Verification
    script_dir = os.path.dirname(__file__)
    if script_dir == '': script_dir = '.'
    
    ckpt_path = os.path.join(script_dir, 'out_looplm', 'ckpt.pt')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(script_dir, 'looplm', 'out_looplm', 'ckpt.pt')
    
    if not os.path.exists(ckpt_path):
        print(f"Error: {ckpt_path} not found.")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_args = checkpoint['model_args']
    state_dict = checkpoint['model']
    
    # 2. Model Initialization
    config = GPTConfig(**model_args)
    model = LoopGPT(config, num_loops=12).to(device).eval()
    
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
    model.load_state_dict(state_dict, strict=False)

    # 3. Data encoding
    meta_paths = [
        os.path.join(script_dir, '..', 'nanoGPT', 'data', 'shakespeare_char', 'meta.pkl'),
        os.path.join(script_dir, 'nanoGPT', 'data', 'shakespeare_char', 'meta.pkl'),
        os.path.join(script_dir, 'data', 'shakespeare_char', 'meta.pkl')
    ]
    
    meta_path = None
    for p in meta_paths:
        if os.path.exists(p):
            meta_path = p
            break
            
    if meta_path is None:
        print("Error: meta.pkl not found.")
        return

    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']

    # 4. Thinking Analysis
    text = "To be, or not to be, that is the question"
    x = torch.tensor([stoi[c] for c in text if c in stoi], dtype=torch.long, device=device)[None, ...]
    
    print("\n--- Phase 2: Thinking Trace Analysis (Pure PyTorch) ---")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Input Text: '{text}'")
    print("Confidence Threshold: 0.9\n")

    with torch.no_grad():
        logits, _, steps = model(x, halt_threshold=0.9)
            
    # 5. Visualization
    steps = steps[0].cpu().numpy()
    chars = [c for c in text if c in stoi]
    
    print("Token-wise Reasoning Depth Visualization:")
    print(f"{'Char':<6} | {'Steps':<6} | {'Visual Trajectory'}")
    print("-" * 45)
    for c, s in zip(chars, steps):
        display_char = '\\n' if c == '\n' else c
        visual = "*" * s
        print(f" '{display_char}'{'' if len(display_char)>1 else ' ':<3} |  {s:2d}    | {visual}")

    avg_depth = steps.mean()
    print("-" * 45)
    print(f"Average Reasoning Depth: {avg_depth:.2f} steps (Max: 12)")
    print(f"Computational Saving: {((12 - avg_depth)/12)*100:.1f}%")

if __name__ == "__main__":
    analyze_thinking_pure_pytorch()
