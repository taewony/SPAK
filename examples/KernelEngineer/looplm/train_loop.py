import os
import time
import math
import pickle
import json
import numpy as np
import torch
from model_loop import LoopGPT
from model import GPTConfig

# -----------------------------------------------------------------------------
# default config values for Addition Experiment
dataset = 'addition_reverse' # Use reverse for better grokking
out_dir = 'looplm/out_addition'
batch_size = 128 
block_size = 64 # Reduced to match typical addition length
n_embd = 256
n_head = 4
num_loops = 12
inject_x0 = False # Disabled for RoPE compatibility as per guide.md
dropout = 0.1
bias = False
learning_rate = 1e-3
max_iters = 15000 # Increased for grokking
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
exec(open(_configurator_path).read()) 
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

script_dir = os.path.dirname(__file__)
if script_dir == '': script_dir = '.'

# Robust out_dir resolution
# If it's a relative path, resolve it relative to the script's directory.
if not os.path.isabs(out_dir):
    full_out_dir = os.path.abspath(os.path.join(script_dir, out_dir))
else:
    full_out_dir = out_dir

os.makedirs(full_out_dir, exist_ok=True)

torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Data loader setup
# 1. Try absolute path relative to script
data_dir = os.path.abspath(os.path.join(script_dir, 'data', dataset))
if not os.path.exists(os.path.join(data_dir, 'train.bin')):
    # 2. Try relative to CWD
    data_dir = os.path.join('data', dataset)
if not os.path.exists(os.path.join(data_dir, 'train.bin')):
    # 3. Try nanoGPT relative path
    data_dir = os.path.join(script_dir, '..', 'nanoGPT', 'data', dataset)

if not os.path.exists(os.path.join(data_dir, 'train.bin')):
    raise FileNotFoundError(f"Could not find training data in {data_dir}")

print(f"Loading data from: {data_dir}")
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

# Pre-calculate newline indices for aligned batching
def get_newline_indices(data):
    # Find all indices where value is 0 (stoi['\n'])
    return np.where(data == 0)[0]

print("Indexing newlines for aligned batching...")
train_newlines = get_newline_indices(train_data)
val_newlines = get_newline_indices(val_data)

def get_batch(split):
    data = train_data if split == 'train' else val_data
    newlines = train_newlines if split == 'train' else val_newlines
    
    # Sample random newline indices as starting points
    # We pick from newlines that have enough room for block_size
    valid_indices = newlines[newlines < len(data) - block_size - 1]
    ix_newlines = torch.randint(len(valid_indices), (batch_size,))
    ix = valid_indices[ix_newlines] + 1 # Start right after \n
    
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# Vocab size from meta
meta_path = os.path.join(data_dir, 'meta.pkl')
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
vocab_size = meta['vocab_size']
stoi = meta['stoi']
thinking_token_id = stoi.get('=', None)

# Model init
model_args = dict(n_layer=1, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=vocab_size, dropout=dropout)
gptconf = GPTConfig(**model_args)
model = LoopGPT(gptconf, num_loops=num_loops, inject_x0=inject_x0)

# Warm Start Logic: Prioritize addition checkpoint, fallback to shakespeare looplm
# Primary: looplm/out_addition/ckpt.pt
init_from_path = os.path.join(script_dir, 'out_addition', 'ckpt.pt')
if not os.path.exists(init_from_path):
    # Fallback 1: looplm/out_looplm/ckpt.pt (Transfer from Shakespeare)
    init_from_path = os.path.join(script_dir, 'out_looplm', 'ckpt.pt')

if os.path.exists(init_from_path):
    print(f"Initializing from existing LoopLM checkpoint: {init_from_path}")
    checkpoint = torch.load(init_from_path, map_location=device)
    state_dict = checkpoint['model']
    
    # Check for vocab mismatch
    ckpt_vocab_size = state_dict['lm_head.weight'].shape[0]
    if ckpt_vocab_size != vocab_size:
        print(f"Vocab mismatch: Ckpt({ckpt_vocab_size}) vs Dataset({vocab_size}).")
        print("Resetting lm_head and wte weights for new task.")
        del state_dict['lm_head.weight']
        del state_dict['transformer.wte.weight']
    
    try:
        model.load_state_dict(state_dict, strict=False)
        print("Successfully loaded weights from checkpoint.")
    except RuntimeError as e:
        print(f"Dimension mismatch or error loading state_dict: {e}")
        print("Starting from scratch instead.")
else:
    print("No previous checkpoint found. Starting from scratch.")

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
        correct = 0
        total = 0
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss, _ = model(X, Y, thinking_token_id=thinking_token_id)
            losses[k] = loss.item()
            
            # Quick Exact Match check for the last token (simple heuristic)
            if k < 20: # Check first 20 batches for accuracy
                preds = torch.argmax(logits, dim=-1) # (B, T) or (B, 1, V)
                # Compare only the last predicted token with the target
                # (Note: In addition, the answer is at the end)
                if Y is not None:
                    # Logic to check if the generated sequence matches target
                    pass
        
        out[split] = losses.mean().item()
    model.train()
    return out

# Improved reporting with sample generation
def report_accuracy(n_samples=10):
    model.eval()
    samples_correct = 0
    for _ in range(n_samples):
        X, Y = get_batch('val')
        # Generate until newline
        idx = X[0:1, :10] # Take a prompt
        # (Simplified for now: just show a sample prediction)
        with torch.no_grad():
            logits, _, _ = model(idx)
            pred = torch.argmax(logits[0, -1, :]).item()
            # print(f"Sample Prompt ID: {idx[0].tolist()} -> Pred ID: {pred}")
    model.train()

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
print(f"Starting LoopLM Training on {dataset}...")
print(f"Config: {num_loops} loops over 1 layer block")
torch.cuda.synchronize()
t0 = time.time()
history = []

for iter_num in range(max_iters):
    # ... (LR logic) ...
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # ... (Eval logic) ...
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
        _, loss, _ = model(X, Y, thinking_token_id=thinking_token_id)
    
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
trace_path = os.path.join(full_out_dir, "looplm_trace.json")
with open(trace_path, "w") as f:
    json.dump(history, f, indent=4)

checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'model_args': model_args,
    'num_loops': num_loops,
    'iter_num': iter_num,
}
torch.save(checkpoint, os.path.join(full_out_dir, 'ckpt.pt'))
print(f"LoopLM Training Complete. Checkpoint: {full_out_dir}/ckpt.pt")
