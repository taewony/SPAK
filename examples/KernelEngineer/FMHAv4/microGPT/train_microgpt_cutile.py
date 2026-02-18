import os
import math
import random
import torch
import torch.nn as nn
from microgpt_cutile import MicroGPT
import time
import json

# --- 1. Data Loading (Identical to microgpt.py) ---
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.seed(42)
random.shuffle(docs)

uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
char_to_idx = {ch: i for i, ch in enumerate(uchars)}

# --- 2. Model Setup (Hyperparams from microgpt.py) ---
n_layer = 1
n_embd = 16
block_size = 16
n_head = 4
device = 'cuda'

model = MicroGPT(vocab_size, n_embd, n_head, n_layer, block_size).to(device).half()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.85, 0.99))

# --- 3. Training Loop ---
num_steps = 100
print(f"Starting cuTile MicroGPT Training for {num_steps} steps...")

history = []
t0 = time.time()

for step in range(num_steps):
    # Take single document (batch size 1 to match microgpt.py exactly)
    doc = docs[step % len(docs)]
    tokens = [BOS] + [char_to_idx[ch] for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)
    
    # Prep tensors
    idx = torch.tensor([tokens[:n]], device=device) # [1, T]
    targets = torch.tensor([tokens[1:n+1]], device=device) # [1, T]
    
    # Forward
    step_start = time.time()
    logits = model(idx) # [1, T, Vocab]
    
    # Loss (Cross Entropy)
    loss = nn.functional.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    step_time = (time.time() - step_start) * 1000
    
    if step % 10 == 0:
        print(f"Step {step} | Loss {loss.item():.4f} | Time {step_time:.2f}ms")
    
    # Emit Trace for SPAK
    trace = {
        "type": "Convergence",
        "step": step,
        "loss": float(loss.item()),
        "step_time_ms": step_time
    }
    history.append(trace)

total_time = time.time() - t0
avg_step = (total_time / num_steps) * 1000

# Final Performance Trace
perf_trace = {
    "type": "Performance",
    "avg_step_time_ms": avg_step,
    "total_time_s": total_time,
    "speedup_vs_baseline": 5.2 / (avg_step) if avg_step > 0 else 0 # 5.2ms is baseline estimate
}
print(f"
__SPAK_TRACE__{json.dumps(perf_trace)}")

# Save history for reflection
with open("microgpt_train_trace.json", "w") as f:
    json.dump(history, f, indent=4)
