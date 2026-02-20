import os
import pickle
import torch
import torch.nn.functional as F
import numpy as np
import random
from model import GPT as GPT_Original, GPTConfig as Config_Original
from nanogpt_cutile import GPT as GPT_CuTile, GPTConfig as Config_CuTile

def set_seed(seed=1337):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def compare_deep():
    ckpt_path = os.path.join(os.path.dirname(__file__), 'out-shakespeare-char', 'ckpt.pt')
    device = 'cuda'
    
    if not os.path.exists(ckpt_path):
        print(f"Error: {ckpt_path} not found.")
        return

    checkpoint = torch.load(ckpt_path, map_location=device)
    model_args = checkpoint['model_args']
    state_dict = checkpoint['model']
    
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    conf_orig = Config_Original(**model_args)
    model_orig = GPT_Original(conf_orig).to(device).eval()
    model_orig.load_state_dict(state_dict)

    conf_ct = Config_CuTile(**model_args)
    model_ct = GPT_CuTile(conf_ct).to(device).eval()
    model_ct.load_state_dict(state_dict)

    T_test = 10
    x = torch.randint(0, model_args['vocab_size'], (1, T_test), device=device)
    
    print(f"--- Deep Block-by-Block Comparison (T={T_test}) ---")
    
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            emb_orig = model_orig.transformer.drop(model_orig.transformer.wte(x) + model_orig.transformer.wpe(torch.arange(T_test, device=device)))
            emb_ct = model_ct.transformer.drop(model_ct.transformer.wte(x) + model_ct.transformer.wpe(torch.arange(T_test, device=device)))
            print(f"Embeddings Diff: {(emb_orig - emb_ct).abs().max().item():.6e}")

            h_orig = emb_orig
            h_ct = emb_ct
            
            for i in range(len(model_orig.transformer.h)):
                h_orig = model_orig.transformer.h[i](h_orig)
                h_ct = model_ct.transformer.h[i](h_ct)
                diff = (h_orig - h_ct).abs().max().item()
                print(f"Block {i} Diff: {diff:.6e}")

            ln_orig = model_orig.transformer.ln_f(h_orig)
            ln_ct = model_ct.transformer.ln_f(h_ct)
            print(f"Final LN Diff: {(ln_orig - ln_ct).abs().max().item():.6e}")
            
            logits_orig = model_orig.lm_head(ln_orig)
            logits_ct = model_ct.lm_head(ln_ct)
            print(f"Final Logits Diff: {(logits_orig - logits_ct).abs().max().item():.6e}")

            print("--- Greedy Sampling (Top-1) Comparison ---")
            set_seed(1337)
            y_orig = model_orig.generate(x[:, :1], 20, temperature=1.0, top_k=1)
            
            set_seed(1337)
            y_ct = model_ct.generate(x[:, :1], 20, temperature=1.0, top_k=1)
            
            meta_path = os.path.join(os.path.dirname(__file__), 'data', 'shakespeare_char', 'meta.pkl')
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            decode = lambda l: ''.join([meta['itos'][i] for i in l])
            
            print(f"Original: {decode(y_orig[0].tolist())}")
            print(f"CuTile:   {decode(y_ct[0].tolist())}")
            
            if y_orig.equal(y_ct):
                print("SUCCESS: Greedy outputs are BIT-IDENTICAL!")
            else:
                print("FAILURE: Outputs diverged.")

if __name__ == "__main__":
    compare_deep()
