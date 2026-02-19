import os
import pickle
import torch
import torch.nn.functional as F
# 두 가지 구현 임포트
from model import GPT as GPT_Original, GPTConfig as Config_Original
from nanogpt_cutile import GPT as GPT_CuTile, GPTConfig as Config_CuTile

# -----------------------------------------------------------------------------
ckpt_path = os.path.join(os.path.dirname(__file__), 'out_nanogpt', 'ckpt.pt')
device = 'cuda'
dtype = 'float16'
# -----------------------------------------------------------------------------

if not os.path.exists(ckpt_path):
    print(f"Error: {ckpt_path} not found.")
    exit()

checkpoint = torch.load(ckpt_path, map_location=device)
model_args = checkpoint['model_args']

# 1. Original 모델 로드
conf_orig = Config_Original(**model_args)
model_orig = GPT_Original(conf_orig).to(device).eval()

# 2. cuTile 모델 로드
conf_cutile = Config_CuTile(**model_args)
model_cutile = GPT_CuTile(conf_cutile).to(device).eval()

# 가중치 이식 (State dict 호환성 확인)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

model_orig.load_state_dict(state_dict)
model_cutile.load_state_dict(state_dict)

# 데이터 인코딩 설정
meta_path = os.path.join(os.path.dirname(__file__), 'data', 'shakespeare_char', 'meta.pkl')
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
decode = lambda l: ''.join([itos[i] for i in l])

start_ids = [stoi['']]
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

print(f"--- Comparison: implementation vs implementation (Same Weights) ---")
with torch.no_grad():
    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        print("[ORIGINAL model.py Output]:")
        y_orig = model_orig.generate(x, 100, temperature=0.8, top_k=200)
        print(decode(y_orig[0].tolist()))
        
        print("" + "="*50)
        
        print("[cuTile nanogpt_cutile.py Output]:")
        y_cutile = model_cutile.generate(x, 100, temperature=0.8, top_k=200)
        print(decode(y_cutile[0].tolist()))
