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
    
    # Bucketized results
    buckets = {5: [0,0], 6: [0,0], 8: [0,0], 10: [0,0], 12: [0,0]} # [correct, total]
    
    # To track thinking depth
    total_steps = 0
    total_tokens_generated = 0
    thinking_token_id = stoi.get('=', None)
    
    for ex in examples:
        if '=' not in ex: continue
        q, a = ex.split('=')
        q_len = len(q.split('+')[0]) # Use length of first operand as bucket
        
        q_input = q + '='
        # Encode question
        x = torch.tensor([stoi[c] for c in q_input], dtype=torch.long, device=device).unsqueeze(0)
        
        # Generate result token by token
        generated = ""
        current_x = x
        max_ans_len = len(a) + 2
        
        with torch.no_grad():
            for _ in range(max_ans_len):
                logits, _, steps = model(current_x, 
                                          halt_threshold=0.9, 
                                          thinking_token_id=thinking_token_id, 
                                          thinking_threshold=0.99)
                last_logits = logits[0, -1, :]
                next_id = torch.argmax(last_logits).item()
                next_char = itos[next_id]
                
                if next_char == '\n': break
                generated += next_char
                total_steps += steps[0, -1].item()
                total_tokens_generated += 1
                
                next_id_tensor = torch.tensor([[next_id]], device=device)
                current_x = torch.cat((current_x, next_id_tensor), dim=1)
                if current_x.size(1) > gptconf.block_size:
                    current_x = current_x[:, 1:]
        
        # After generation
        is_correct = generated.strip() == a.strip()
        if is_correct: correct += 1
        total += 1
        
        # Track buckets
        for b_size in sorted(buckets.keys(), reverse=True):
            if q_len >= b_size:
                buckets[b_size][1] += 1
                if is_correct: buckets[b_size][0] += 1
                break

    accuracy = correct / total if total > 0 else 0
    avg_steps = total_steps / total_tokens_generated if total_tokens_generated > 0 else 0
    
    print(f"\n--- OOD Detailed Report ---")
    for b_size, counts in buckets.items():
        if counts[1] > 0:
            print(f"  {b_size}+ Digits: {counts[0]/counts[1]*100:5.2f}% ({counts[0]}/{counts[1]})")
    print(f"Overall OOD Accuracy: {accuracy*100:.2f}%")
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
