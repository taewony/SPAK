import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from nanogpt_cutile import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume' # 'resume' or 'gpt2*'
out_dir = 'out_nanogpt'
start = "\n" 
num_samples = 5

max_new_tokens = 100
temperature = 0.8
top_k = 200
device = 'cuda'
dtype = 'float16' # Use float16 for cuTile
# -----------------------------------------------------------------------------

torch.manual_seed(1337)
torch.cuda.manual_seed(1337)
device_type = 'cuda'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    if not os.path.exists(ckpt_path):
        # Fallback to current best if available
        print(f"No checkpoint found at {ckpt_path}. Checking local directory...")
        ckpt_path = 'ckpt.pt'
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    model = GPT.from_pretrained(init_from)

model.eval()
model.to(device)

# Encoder/Decoder setup
load_meta = False
if init_from == 'resume':
    # Try to find meta.pkl from data/dataset/
    # For shakespeare_char, it's usually there
    meta_path = os.path.join(os.path.dirname(__file__), 'data', 'shakespeare_char', 'meta.pkl')
    load_meta = os.path.exists(meta_path)

if load_meta:
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    print("Assuming GPT-2 BPE encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the prompt
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
print(f"--- Sampling from {init_from} using cuTile Kernels ---")
inference_results = []
import time
import json

with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            t0 = time.time()
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            t1 = time.time()
            
            generated_text = decode(y[0].tolist())
            dt = t1 - t0
            tokens_per_sec = max_new_tokens / dt
            
            print(generated_text)
            print('---------------')
            
            inference_results.append({
                "sample_id": k,
                "text": generated_text,
                "time_s": dt,
                "tokens_per_sec": tokens_per_sec
            })

# Save Inference Trace
with open("nanogpt_inference_trace.json", "w") as f:
    json.dump(inference_results, f, indent=4)

print(f"\nInference complete. Results saved to nanogpt_inference_trace.json")
