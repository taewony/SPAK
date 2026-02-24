import os
import sys
import time
import math
import pickle
import json
import numpy as np
import torch
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values
dataset = 'addition_reverse'
out_dir = 'looplm/out_baseline_12l'
batch_size = 128 
block_size = 64
n_layer = 12
n_head = 4
n_embd = 256
dropout = 0.1
bias = False
learning_rate = 1e-3
max_iters = 15000
eval_interval = 500
eval_iters = 200
log_interval = 100
device = 'cuda'
dtype = 'float16' 
decay_lr = True
warmup_iters = 100
lr_decay_iters = 15000
min_lr = 1e-4
beta2 = 0.99
weight_decay = 1e-4
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
# Robust path resolution for configurator.py
_configurator_path = os.path.join(os.path.dirname(__file__), 'configurator.py')
if not os.path.exists(_configurator_path):
    _configurator_path = 'configurator.py' # Fallback

# Set default config if no arguments provided
if len(sys.argv) == 1:
    default_config = os.path.join(os.path.dirname(__file__), 'config', 'train_12l_baseline.py')
    if os.path.exists(default_config):
        sys.argv.append(default_config)

exec(open(_configurator_path).read()) 
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# Set paths relative to script location
script_dir = os.path.dirname(__file__)
if script_dir == '': script_dir = '.'
full_out_dir = out_dir # If starting with looplm/, it's already relative to root

os.makedirs(full_out_dir, exist_ok=True)

torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Data loader setup
# We expect data to be in nanoGPT/data/shakespeare_char/
# script_dir is either '.' or the path to looplm/
data_dir = os.path.join(os.path.dirname(script_dir), 'nanoGPT', 'data', dataset)
if not os.path.exists(data_dir):
    # Try direct project root if nested
    data_dir = os.path.join('..', 'nanoGPT', 'data', dataset)
if not os.path.exists(data_dir):
    data_dir = os.path.join('data', dataset)
if not os.path.exists(data_dir):
    # Fallback to local data dir if all else fails
    data_dir = os.path.join(script_dir, 'data', dataset)

train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

# Pre-calculate newline indices for aligned batching
def get_newline_indices(data):
    return np.where(data == 0)[0]

print("Indexing newlines for aligned batching...")
train_newlines = get_newline_indices(train_data)
val_newlines = get_newline_indices(val_data)

def get_batch(split):
    data = train_data if split == 'train' else val_data
    newlines = train_newlines if split == 'train' else val_newlines
    valid_indices = newlines[newlines < len(data) - block_size - 1]
    ix_newlines = torch.randint(len(valid_indices), (batch_size,))
    ix = valid_indices[ix_newlines] + 1
    
    x_stack = []
    y_stack = []
    
    for i in ix:
        chunk_x = torch.from_numpy((data[i:i+block_size]).astype(np.int64))
        chunk_y = torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64))
        
        target_mask = torch.full_like(chunk_y, -1)
        eq_indices = (chunk_x == thinking_token_id).nonzero(as_tuple=True)[0]
        nl_indices = (chunk_x == 0).nonzero(as_tuple=True)[0]
        
        sample_starts = torch.cat((torch.tensor([0], device=chunk_x.device), nl_indices + 1))
        
        for start in sample_starts:
            if start >= block_size: break
            future_eqs = eq_indices[eq_indices >= start]
            if len(future_eqs) == 0: continue
            eq_pos = future_eqs[0]
            future_nls = nl_indices[nl_indices > eq_pos]
            end_pos = future_nls[0] if len(future_nls) > 0 else torch.tensor(block_size, device=chunk_x.device)
            target_mask[eq_pos:end_pos] = chunk_y[eq_pos:end_pos]
            
        x_stack.append(chunk_x)
        y_stack.append(target_mask)
        
    x = torch.stack(x_stack).to(device)
    y = torch.stack(y_stack).to(device)
    return x, y

# Vocab size from meta
meta_path = os.path.join(data_dir, 'meta.pkl')
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
vocab_size = meta['vocab_size']
stoi = meta['stoi']
thinking_token_id = stoi.get('=', None)

# Model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=vocab_size, dropout=dropout)
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
model.to(device)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, beta2), weight_decay=weight_decay)
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
                _, loss = model(X, Y, thinking_token_id=thinking_token_id)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# Training Loop
print(f"Starting 12L Baseline Training on {dataset}...")
torch.cuda.synchronize()
t0 = time.time()
history = []

for iter_num in range(max_iters):
    # ... (LR logic) ...
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr:.4e}")
        history.append({
            "type": "Convergence",
            "step": iter_num,
            "train_loss": losses['train'],
            "val_loss": losses['val']
        })

    X, Y = get_batch('train')
    with ctx:
        _, loss = model(X, Y, thinking_token_id=thinking_token_id)
    
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    if iter_num % log_interval == 0:
        torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1 - t0) * 1000 / log_interval if iter_num > 0 else (t1 - t0) * 1000
        t0 = t1
        print(f"iter {iter_num}: loss {loss.item():.4f}, time {dt:.2f}ms")
        if iter_num % 100 == 0:
            history.append({
                "type": "Performance",
                "step": iter_num,
                "loss": loss.item(),
                "step_time_ms": dt
            })

# Save trace
trace_path = os.path.join(full_out_dir, "baseline_12l_trace.json")
with open(trace_path, "w") as f:
    json.dump(history, f, indent=4)

checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'model_args': model_args,
    'iter_num': iter_num,
}
torch.save(checkpoint, os.path.join(full_out_dir, 'ckpt.pt'))
print(f"\n12L Baseline Training Complete. Checkpoint: {full_out_dir}/ckpt.pt")
