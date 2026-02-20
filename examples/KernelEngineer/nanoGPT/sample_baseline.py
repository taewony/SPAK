import os
import pickle
import torch
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
out_dir = 'out_baseline' # 'out_baseline' folder is assumed to be in the same dir as this script
device = 'cuda'
dtype = 'float16'
# -----------------------------------------------------------------------------

# Get the directory where this script is located
script_dir = os.path.dirname(__file__)

torch.manual_seed(1337)
device_type = 'cuda'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

ckpt_path = os.path.join(script_dir, out_dir, 'ckpt.pt')
if not os.path.exists(ckpt_path):
    print(f"No checkpoint found at {ckpt_path}. Please run train_pytorch_baseline.py first.")
    exit()

checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()
model.to(device)

# Encoder/Decoder setup
# Shakespeare char meta.pkl location
meta_path = os.path.join(script_dir, 'data', 'shakespeare_char', 'meta.pkl')
if not os.path.exists(meta_path):
    # Fallback to root-relative if not found in nanoGPT/data
    meta_path = os.path.join(script_dir, '..', 'data', 'shakespeare_char', 'meta.pkl')

with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# encode the prompt
start = "\n"
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
print(f"--- Sampling from Baseline (PyTorch) ---")
with torch.no_grad():
    with torch.amp.autocast(device_type=device_type, dtype=ptdtype):
        y = model.generate(x, 100, temperature=0.8, top_k=200)
        print(decode(y[0].tolist()))
        print('---------------')
