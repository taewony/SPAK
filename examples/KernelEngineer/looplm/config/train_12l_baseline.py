# train a 12-layer GPT baseline for LoopLM comparison
# target: Val Loss ~1.47

out_dir = 'looplm/out_baseline_12l'
eval_interval = 250
eval_iters = 200
log_interval = 10

always_save_checkpoint = False

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

# Standard GPT-2 Small Config (12L, 12H, 768D)
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.2

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100
