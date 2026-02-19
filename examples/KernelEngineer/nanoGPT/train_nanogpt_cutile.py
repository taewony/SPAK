import os
import time
import math
import pickle
import json
import numpy as np
import torch
from nanogpt_cutile import GPTConfig, GPT

# -----------------------------------------------------------------------------
# Configuration (Optimized for rapid experiment on shakespeare_char)
dataset = 'shakespeare_char'
out_dir = 'out_nanogpt'
batch_size = 12
block_size = 256
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.0
bias = False
learning_rate = 1e-3
max_iters = 500
eval_interval = 100
eval_iters = 20
device = 'cuda'
dtype = 'float16' # Use float16 for cuTile performance testing
# -----------------------------------------------------------------------------

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Data loader
data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# Vocab size from meta
meta_path = os.path.join(data_dir, 'meta.pkl')
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
vocab_size = meta['vocab_size']

# Model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=vocab_size, dropout=dropout)
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
model.to(device)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=1e-1)
scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# Training Loop
print(f"Starting NanoGPT cuTile Training on {dataset}...")
t0 = time.time()
history = []

for iter_num in range(max_iters):
    # Eval
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        history.append({
            "type": "Convergence",
            "step": iter_num,
            "train_loss": losses['train'],
            "val_loss": losses['val']
        })

    # Step
    X, Y = get_batch('train')
    with ctx:
        _, loss = model(X, Y)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    if iter_num % 10 == 0:
        t1 = time.time()
        dt = (t1 - t0) * 1000 / 10 if iter_num > 0 else (t1 - t0) * 1000
        t0 = t1
        print(f"iter {iter_num}: loss {loss.item():.4f}, time {dt:.2f}ms")

# Save Trace
with open("nanogpt_train_trace.json", "w") as f:
    json.dump(history, f, indent=4)

# --- Save Checkpoint ---
checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'model_args': model_args,
    'iter_num': iter_num,
}
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
torch.save(checkpoint, ckpt_path)
print(f"Checkpoint saved to {ckpt_path}")

print("\nTraining Complete. Trace saved to nanogpt_train_trace.json")
