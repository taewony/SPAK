import os
import pickle
import torch
import numpy as np
import json
from model_loop import LoopGPT
from model import GPTConfig, GPT

def count_carries(n1, n2):
    s1, s2 = str(n1)[::-1], str(n2)[::-1]
    max_l = max(len(s1), len(s2))
    s1, s2 = s1.ljust(max_l, '0'), s2.ljust(max_l, '0')
    carries, current_carry = 0, 0
    for i in range(max_l):
        digit_sum = int(s1[i]) + int(s2[i]) + current_carry
        if digit_sum >= 10:
            carries += 1
            current_carry = 1
        else:
            current_carry = 0
    return carries

def evaluate_ood(ckpt_path, device='cuda', num_samples=100, max_loops=None):
    print(f"Evaluating OOD for {ckpt_path}...")
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint {ckpt_path} not found.")
        return None

    checkpoint = torch.load(ckpt_path, map_location=device)
    config_dict = checkpoint.get('config', {})
    dataset = config_dict.get('dataset', 'addition')
    if 'addition_reverse' in ckpt_path: dataset = 'addition_reverse'

    args = checkpoint['model_args']
    gptconf = GPTConfig(**args)
    
    is_loop = 'num_loops' in checkpoint or 'num_loops' in config_dict
    if is_loop:
        n_loops = max_loops if max_loops is not None else checkpoint.get('num_loops', 12)
        inject_x0 = config_dict.get('inject_x0', False) # Default to False for RoPE
        model = LoopGPT(gptconf, num_loops=n_loops, inject_x0=inject_x0)
    else:
        model = GPT(gptconf)
    
    model.load_state_dict(checkpoint['model'], strict=False)
    model.to(device)
    model.eval()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, 'data', dataset))
    meta_path = os.path.join(data_dir, 'meta.pkl')
    if not os.path.exists(meta_path):
        data_dir = os.path.join('data', dataset)
        meta_path = os.path.join(data_dir, 'meta.pkl')
        
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    itos, stoi = meta['itos'], meta['stoi']

    ood_bin_path = os.path.join(data_dir, 'val_ood.bin')
    ood_data = np.memmap(ood_bin_path, dtype=np.uint16, mode='r')
    text = "".join([itos[i] for i in ood_data])
    examples = text.strip().split('\n')
    
    if num_samples > 0:
        import random
        random.seed(42) # Reproducibility
        examples = random.sample(examples, min(num_samples, len(examples)))

    correct = 0
    total = 0
    buckets = {1: [0,0,0,0], 5: [0,0,0,0], 6: [0,0,0,0], 8: [0,0,0,0], 10: [0,0,0,0], 12: [0,0,0,0]}
    carry_stats = {} # [correct, total]
    total_steps, total_tokens_generated = 0, 0
    thinking_token_id = stoi.get('=', None)
    
    for ex in examples:
        if '=' not in ex: continue
        q, a = ex.split('=')
        parts = q.split('+')
        q_len = max(len(parts[0]), len(parts[1]))
        
        # Parse operands for carry diagnostic
        try:
            if 'addition_reverse' in dataset:
                n1, n2 = int(parts[0][::-1]), int(parts[1][::-1])
            else:
                n1, n2 = int(parts[0]), int(parts[1])
            num_c = count_carries(n1, n2)
        except: num_c = 0
        
        q_input = q + '='
        x = torch.tensor([stoi[c] for c in q_input], dtype=torch.long, device=device).unsqueeze(0)
        generated, current_x = "", x
        sample_steps, sample_tokens = 0, 0
        
        with torch.no_grad():
            for _ in range(len(a) + 2):
                if is_loop:
                    # [CRITICAL FIX] 
                    # Set thresholds to None to DISABLE dynamic halting.
                    # This forces the model to utilize all its learned reasoning depth.
                    logits, _, steps = model(current_x, halt_threshold=None, thinking_threshold=None)
                    steps_taken_val = n_loops # Full loops used
                else:
                    logits, _ = model(current_x)
                    steps_taken_val = gptconf.n_layer
                
                next_id = torch.argmax(logits[0, -1, :]).item()
                next_char = itos[next_id]
                if next_char == '\n': break
                
                generated += next_char
                sample_steps += steps_taken_val
                sample_tokens += 1
                current_x = torch.cat((current_x, torch.tensor([[next_id]], device=device)), dim=1)
                if current_x.size(1) > gptconf.block_size: current_x = current_x[:, 1:]
        
        is_correct = generated.strip() == a.strip()
        if is_correct: correct += 1
        total += 1
        total_steps += sample_steps
        total_tokens_generated += sample_tokens
        
        if num_c not in carry_stats: carry_stats[num_c] = [0, 0]
        carry_stats[num_c][1] += 1
        if is_correct: carry_stats[num_c][0] += 1
        
        for b_size in sorted(buckets.keys(), reverse=True):
            if q_len >= b_size:
                buckets[b_size][1] += 1
                if is_correct: buckets[b_size][0] += 1
                buckets[b_size][2] += sample_steps
                buckets[b_size][3] += sample_tokens
                break

    print(f"\n--- Carry-wise Diagnostic Report ---")
    for c in sorted(carry_stats.keys()):
        stats = carry_stats[c]
        print(f"{c:2d} Carries | {(stats[0]/stats[1]*100):8.2f}% ({stats[0]}/{stats[1]})")

    print(f"\n--- OOD Detailed Intelligence Report ---")
    for b_size, data in sorted(buckets.items()):
        if data[1] > 0:
            acc = (data[0]/data[1]) * 100
            avg_s = data[2]/data[3] if data[3] > 0 else 0
            label = f"{b_size:2d}+ Digits" if b_size > 1 else "1-4 Digits"
            print(f"{label:<10} | {acc:8.2f}% | {avg_s:9.2f}")
    
    overall_avg_steps = total_steps / total_tokens_generated if total_tokens_generated > 0 else 0
    return {
        "accuracy": correct/total if total>0 else 0, 
        "correct": correct, 
        "total": total, 
        "avg_steps": overall_avg_steps, 
        "buckets": buckets,
        "carry_stats": carry_stats # Added for diagnostic storage
    }

if __name__ == "__main__":
    import sys
    ckpt = sys.argv[1] if len(sys.argv) > 1 else 'looplm/out_addition/ckpt.pt'
    evaluate_ood(ckpt)
