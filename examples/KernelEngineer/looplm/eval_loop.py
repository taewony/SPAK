import os
import pickle
import torch
import numpy as np
import json
from model_loop import LoopGPT
from model import GPTConfig, GPT

def evaluate_ood(ckpt_path, device='cuda', num_samples=100, max_loops=None):
    print(f"Evaluating OOD for {ckpt_path} (n={num_samples}, max_loops={max_loops})...")
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
        inject_x0 = config_dict.get('inject_x0', True)
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
        random.seed(42)
        examples = random.sample(examples, min(num_samples, len(examples)))

    correct = 0
    total = 0
    buckets = {5: [0,0,0,0], 6: [0,0,0,0], 8: [0,0,0,0], 10: [0,0,0,0], 12: [0,0,0,0]}
    total_steps, total_tokens_generated = 0, 0
    thinking_token_id = stoi.get('=', None)
    
    for ex in examples:
        if '=' not in ex: continue
        q, a = ex.split('=')
        q_len = len(q.split('+')[0])
        
        q_input = q + '='
        x = torch.tensor([stoi[c] for c in q_input], dtype=torch.long, device=device).unsqueeze(0)
        
        generated = ""
        current_x = x
        max_ans_len = len(a) + 2
        sample_steps = 0
        sample_tokens = 0
        
        with torch.no_grad():
            for _ in range(max_ans_len):
                if is_loop:
                    logits, _, steps = model(current_x, halt_threshold=0.9, 
                                              thinking_token_id=thinking_token_id, thinking_threshold=0.99)
                    steps_taken_val = steps[0, -1].item()
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
        
        for b_size in sorted(buckets.keys(), reverse=True):
            if q_len >= b_size:
                buckets[b_size][1] += 1
                if is_correct: buckets[b_size][0] += 1
                buckets[b_size][2] += sample_steps
                buckets[b_size][3] += sample_tokens
                break

    accuracy = correct / total if total > 0 else 0
    print(f"\n--- OOD Detailed Intelligence Report ---")
    print(f"{'Digits':<10} | {'Accuracy':<10} | {'Avg Steps':<10}")
    print("-" * 35)
    for b_size, data in buckets.items():
        if data[1] > 0:
            acc = (data[0]/data[1]) * 100
            avg_s = data[2]/data[3] if data[3] > 0 else 0
            print(f"{b_size:2d}+ Digits | {acc:8.2f}% | {avg_s:9.2f}")
    
    overall_avg_steps = total_steps / total_tokens_generated if total_tokens_generated > 0 else 0
    print(f"\nOverall OOD Accuracy: {accuracy*100:.2f}%")
    return {"accuracy": accuracy, "correct": correct, "total": total, "avg_steps": overall_avg_steps}

if __name__ == "__main__":
    import sys
    ckpt = sys.argv[1] if len(sys.argv) > 1 else 'looplm/out_addition/ckpt.pt'
    evaluate_ood(ckpt)
