import os
import pickle
import torch
import torch.nn.functional as F
from model_loop import LoopGPT
from model import GPTConfig

def analyze_thinking_addition():
    script_dir = os.path.dirname(__file__)
    if script_dir == '': script_dir = '.'
    
    # 1. Prioritize out_addition checkpoint
    ckpt_path = os.path.join(script_dir, 'out_addition', 'ckpt.pt')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(script_dir, 'looplm', 'out_addition', 'ckpt.pt')
    
    if not os.path.exists(ckpt_path):
        print(f"Error: {ckpt_path} not found. Please train LoopLM on addition dataset first.")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_args = checkpoint['model_args']
    state_dict = checkpoint['model']
    
    config = GPTConfig(**model_args)
    model = LoopGPT(config, num_loops=12).to(device).eval()
    
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)

    # 2. Load Addition Meta
    meta_path = os.path.join(script_dir, 'data', 'addition', 'meta.pkl')
    if not os.path.exists(meta_path):
        meta_path = os.path.join(script_dir, 'looplm', 'data', 'addition', 'meta.pkl')
        
    if not os.path.exists(meta_path):
        print(f"Error: {meta_path} not found. Run addition_prepare.py first.")
        return

    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']

    # 3. Addition Prompt
    text = "123+456="
    x = torch.tensor([stoi[c] for c in text if c in stoi], dtype=torch.long, device=device)[None, ...]
    
    print("--- Phase 2: Thinking Trace Analysis (ADDITION) ---")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Input: '{text}'")
    print("Confidence Threshold: 0.9")

    with torch.no_grad():
        logits, _, steps = model(x, halt_threshold=0.9)
            
    # 4. Visualize
    steps = steps[0].cpu().numpy()
    chars = [c for c in text if c in stoi]
    
    print("Token-wise Reasoning Depth (Addition carry logic):")
    print("Char | Steps | Visual")
    print("-" * 30)
    for c, s in zip(chars, steps):
        visual = "*" * s
        print(f" '{c}'  |  {s:2d}   | {visual}")

    avg_depth = steps.mean()
    print(f"Average Reasoning Depth: {avg_depth:.2f} iterations")
    print(f"Theoretical Efficiency Gain: {((12 - avg_depth)/12)*100:.1f}%")

if __name__ == "__main__":
    analyze_thinking_addition()
