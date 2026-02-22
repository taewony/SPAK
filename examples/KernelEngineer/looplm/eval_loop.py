import os
import pickle
import torch
import numpy as np
import json
from model_loop import LoopGPT
from model import GPTConfig

def evaluate_ood(ckpt_path, device='cuda', num_samples=100, max_loops=None):
    print(f"Evaluating OOD for {ckpt_path} (n={num_samples}, max_loops={max_loops})...")
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint {ckpt_path} not found.")
        return None

    checkpoint = torch.load(ckpt_path, map_location=device)
    # The saved model_args might not include everything, ensure it's compatible
    args = checkpoint['model_args']
    gptconf = GPTConfig(**args)
    
    n_loops = max_loops if max_loops is not None else checkpoint['num_loops']
    model = LoopGPT(gptconf, num_loops=n_loops)
    
    # Check for strict=False because of potential minor state dict differences
    model.load_state_dict(checkpoint['model'], strict=False)
    model.to(device)
    model.eval()

    # Load meta and OOD data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data', 'addition')
    meta_path = os.path.join(data_dir, 'meta.pkl')
    if not os.path.exists(meta_path):
        # Try relative to CWD if script_dir fails
        data_dir = 'looplm/data/addition'
        meta_path = os.path.join(data_dir, 'meta.pkl')
        
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    itos = meta['itos']
    stoi = meta['stoi']

    ood_bin_path = os.path.join(data_dir, 'val_ood.bin')
    ood_data = np.memmap(ood_bin_path, dtype=np.uint16, mode='r')
    text = "".join([itos[i] for i in ood_data])
    examples = text.strip().split('\n')
    
    if num_samples > 0:
        import random
        random.seed(42)
        examples = random.sample(examples, min(num_samples, len(examples)))

    correct = 0
    total = 0
    
    # To track thinking depth
    total_steps = 0
    total_tokens_generated = 0
    thinking_token_id = stoi.get('=', None)
    
    for ex in examples:
        if '=' not in ex: continue
        q, a = ex.split('=')
        q = q + '='
        
        # Encode question
        x = torch.tensor([stoi[c] for c in q], dtype=torch.long, device=device).unsqueeze(0)
        
        # Generate result token by token
        generated = ""
        current_x = x
        
        # Max length for answer: length of correct answer + some buffer
        max_ans_len = len(a) + 2
        
        with torch.no_grad():
            for _ in range(max_ans_len):
                # Using thinking_threshold=0.99 for output part
                logits, _, steps = model(current_x, 
                                          halt_threshold=0.9, 
                                          thinking_token_id=thinking_token_id, 
                                          thinking_threshold=0.99)
                # logits is (B, 1, V) because we are in eval mode (optimized for last token)
                last_logits = logits[0, -1, :]
                probs = torch.softmax(last_logits, dim=-1)
                next_id = torch.argmax(probs).item()
                next_char = itos[next_id]
                
                if next_char == '\n': break
                generated += next_char
                # steps is (B, T), take only the last token's steps
                total_steps += steps[0, -1].item()
                total_tokens_generated += 1
                
                # Append to input for next step
                next_id_tensor = torch.tensor([[next_id]], device=device)
                current_x = torch.cat((current_x, next_id_tensor), dim=1)
                if current_x.size(1) > gptconf.block_size:
                    current_x = current_x[:, 1:]
        
        if generated.strip() == a.strip():
            correct += 1
        total += 1
        
    accuracy = correct / total if total > 0 else 0
    avg_steps = total_steps / total_tokens_generated if total_tokens_generated > 0 else 0
    
    print(f"OOD Accuracy: {accuracy*100:.2f}% ({correct}/{total})")
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "avg_steps": avg_steps
    }

if __name__ == "__main__":
    import sys
    ckpt = sys.argv[1] if len(sys.argv) > 1 else 'looplm/out_addition/ckpt.pt'
    evaluate_ood(ckpt)
