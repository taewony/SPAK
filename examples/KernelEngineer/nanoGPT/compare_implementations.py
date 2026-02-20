import os
import pickle
import torch
from model import GPT as GPT_Original, GPTConfig as Config_Original
from nanogpt_cutile import GPT as GPT_CuTile, GPTConfig as Config_CuTile

# -----------------------------------------------------------------------------
out_dirs = ['out_nanogpt', 'out-shakespeare-char', 'out_baseline']
device = 'cuda'
dtype = 'float16'
# -----------------------------------------------------------------------------

# Meta 정보 로드
meta_path = os.path.join(os.path.dirname(__file__), 'data', 'shakespeare_char', 'meta.pkl')
if not os.path.exists(meta_path):
    print(f"Error: {meta_path} not found. Please run prepare.py first.")
    exit()

with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
decode = lambda l: ''.join([itos[i] for i in l])

# 시작 토큰 설정 (공백 또는 줄바꿈)
start_char = '\n'
x = torch.tensor([stoi[start_char]], dtype=torch.long, device=device)[None, ...]

def run_inference(out_dir):
    ckpt_path = os.path.join(os.path.dirname(__file__), out_dir, 'ckpt.pt')
    if not os.path.exists(ckpt_path):
        print(f"\n[SKIP] {out_dir}: Checkpoint not found.")
        return

    print(f"\n" + "="*60)
    print(f"--- FOLDER: {out_dir} ---")
    print(f"="*60)

    checkpoint = torch.load(ckpt_path, map_location=device)
    model_args = checkpoint['model_args']
    state_dict = checkpoint['model']
    
    # Prefix 제거
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    # 두 모델 로드
    m_orig = GPT_Original(Config_Original(**model_args)).to(device).eval()
    m_orig.load_state_dict(state_dict)

    m_ct = GPT_CuTile(Config_CuTile(**model_args)).to(device).eval()
    m_ct.load_state_dict(state_dict)

    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            # 1. Numerical Check (Block 0)
            t_in = x.clone()
            h_orig = m_orig.transformer.drop(m_orig.transformer.wte(t_in) + m_orig.transformer.wpe(torch.arange(1, device=device)))
            h_orig = m_orig.transformer.h[0](h_orig)
            
            h_ct = m_ct.transformer.drop(m_ct.transformer.wte(t_in) + m_ct.transformer.wpe(torch.arange(1, device=device)))
            h_ct = m_ct.transformer.h[0](h_ct)
            diff = (h_orig - h_ct).abs().max().item()
            print(f"[Numerical] Max Diff after Block 0: {diff:.6e}")

            # 2. Sample from Original
            print(f"\n[Inference: ORIGINAL model.py]")
            y_orig = m_orig.generate(x, 100, temperature=0.8, top_k=200)
            print("-" * 20)
            print(decode(y_orig[0].tolist()))
            print("-" * 20)

            # 3. Sample from cuTile
            print(f"\n[Inference: cuTile nanogpt_cutile.py]")
            y_ct = m_ct.generate(x, 100, temperature=0.8, top_k=200)
            print("-" * 20)
            print(decode(y_ct[0].tolist()))
            print("-" * 20)

if __name__ == "__main__":
    for d in out_dirs:
        run_inference(d)
