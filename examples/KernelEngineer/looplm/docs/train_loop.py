# =============================================================================
# train_loop.py – Semantic Norm / Pseudo Code Representation
# =============================================================================
#
# This file contains the training loop configuration and execution logic for
# LoopLM on the addition dataset.
# =============================================================================

import os, time, math, pickle, json
import numpy as np
import torch
from model_loop import LoopGPT
from model import GPTConfig

# -----------------------------------------------------------------------------
# Configuration (default values, can be overridden by configurator.py)
# -----------------------------------------------------------------------------
dataset = 'addition'
out_dir = 'looplm/out_addition'
batch_size = 64
block_size = 256
n_embd = 384
n_head = 6
num_loops = 12
dropout = 0.2
bias = False
learning_rate = 1e-3
max_iters = 2000
eval_interval = 250
eval_iters = 200
log_interval = 10
device = 'cuda'
dtype = 'float16'
decay_lr = True
warmup_iters = 100
lr_decay_iters = 2000
min_lr = 1e-4
beta2 = 0.99

# -----------------------------------------------------------------------------
# Configurator override (if configurator.py exists, it updates globals)
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
_configurator_path = os.path.join(os.path.dirname(__file__), 'configurator.py')
if os.path.exists(_configurator_path):
    exec(open(_configurator_path).read())
config = {k: globals()[k] for k in config_keys}

# -----------------------------------------------------------------------------
# Output directory setup
# -----------------------------------------------------------------------------
script_dir = os.path.dirname(__file__) or '.'
if not out_dir.startswith(script_dir):
    full_out_dir = os.path.join(script_dir, os.path.basename(out_dir))
else:
    full_out_dir = out_dir
os.makedirs(full_out_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# Reproducibility and precision settings
# -----------------------------------------------------------------------------
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# -----------------------------------------------------------------------------
# Data loader (using memory-mapped numpy files from nanoGPT format)
# -----------------------------------------------------------------------------
data_dir = os.path.join(script_dir, 'data', dataset)
if not os.path.exists(data_dir):
    data_dir = os.path.join(script_dir, '..', 'nanoGPT', 'data', dataset)

train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data   = np.memmap(os.path.join(data_dir, 'val.bin'),   dtype=np.uint16, mode='r')

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

# -----------------------------------------------------------------------------
# Vocabulary size from meta.pkl
# -----------------------------------------------------------------------------
meta_path = os.path.join(data_dir, 'meta.pkl')
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
vocab_size = meta['vocab_size']

# -----------------------------------------------------------------------------
# Model creation
# -----------------------------------------------------------------------------
model_args = dict(
    n_layer=1, n_head=n_head, n_embd=n_embd, block_size=block_size,
    bias=bias, vocab_size=vocab_size, dropout=dropout
)
gptconf = GPTConfig(**model_args)
model = LoopGPT(gptconf, num_loops=num_loops)

# -----------------------------------------------------------------------------
# Warm start (checkpoint loading)
# -----------------------------------------------------------------------------
init_from_path = os.path.join(script_dir, 'out_addition', 'ckpt.pt')
if not os.path.exists(init_from_path):
    init_from_path = os.path.join(script_dir, 'out_looplm', 'ckpt.pt')

if os.path.exists(init_from_path):
    checkpoint = torch.load(init_from_path, map_location=device)
    state_dict = checkpoint['model']
    ckpt_vocab_size = state_dict['lm_head.weight'].shape[0]
    if ckpt_vocab_size != vocab_size:
        # reset head and embedding for new vocabulary
        del state_dict['lm_head.weight']
        del state_dict['transformer.wte.weight']
    model.load_state_dict(state_dict, strict=False)

model.to(device)

# -----------------------------------------------------------------------------
# Optimizer and gradient scaler
# -----------------------------------------------------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                              betas=(0.9, beta2), weight_decay=1e-1)
scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))

# -----------------------------------------------------------------------------
# Helper: estimate loss on train/val splits
# -----------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                _, loss, _ = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# -----------------------------------------------------------------------------
# Learning rate scheduler (cosine decay with warmup)
# -----------------------------------------------------------------------------
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------
t0 = time.time()
history = []

for iter_num in range(max_iters):
    # update learning rate
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluation
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr:.4e}")
        history.append({"type": "Convergence", "step": iter_num,
                        "train_loss": losses['train'], "val_loss": losses['val']})

    # get a batch
    X, Y = get_batch('train')

    # forward + backward with mixed precision
    with ctx:
        _, loss, _ = model(X, Y)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # logging
    if iter_num % log_interval == 0:
        t1 = time.time()
        dt = (t1 - t0) * 1000 / log_interval if iter_num > 0 else (t1 - t0) * 1000
        t0 = t1
        print(f"iter {iter_num}: loss {loss.item():.4f}, time {dt:.2f}ms")
        if iter_num % 100 == 0:
            history.append({"type": "Performance", "step": iter_num,
                            "loss": loss.item(), "step_time_ms": dt})

# -----------------------------------------------------------------------------
# Save trace and checkpoint
# -----------------------------------------------------------------------------
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
