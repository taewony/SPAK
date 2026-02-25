import os
import pickle
import torch
import numpy as np
from model_loop import LoopGPT
from model import GPTConfig, GPT


def evaluate_algebra_v2(ckpt_path, device='cuda', num_samples=200):
    print(f"Evaluating Algebra V2 for {ckpt_path}...")
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint {ckpt_path} not found.")
        return

    checkpoint = torch.load(ckpt_path, map_location=device)
    config_dict = checkpoint.get('config', {})
    dataset = 'algebra_reverse' # Match addition_algebra_reverse_prepare.py

    args = checkpoint['model_args']
    gptconf = GPTConfig(**args)

    is_loop = 'num_loops' in checkpoint or 'num_loops' in config_dict
    if is_loop:
        n_loops = checkpoint.get('num_loops', 20)
        inject_x0 = config_dict.get('inject_x0', False)
        model = LoopGPT(gptconf, num_loops=n_loops, inject_x0=inject_x0)
    else:
        model = GPT(gptconf)

    model.load_state_dict(checkpoint['model'], strict=False)
    model.to(device)
    model.eval()

    # -------------------------
    # Load dataset
    # -------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, 'data', dataset))
    meta_path = os.path.join(data_dir, 'meta.pkl')

    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    itos, stoi = meta['itos'], meta['stoi']

    ood_bin_path = os.path.join(data_dir, 'val_ood.bin')
    ood_data = np.memmap(ood_bin_path, dtype=np.uint16, mode='r')
    text = "".join([itos[i] for i in ood_data])
    examples = text.strip().split('\n')

    if num_samples > 0:
        import random
        random.seed(42)
        examples = random.sample(examples, min(num_samples, len(examples)))

    correct = {"N": 0, "A": 0, "B": 0}
    total = {"N": 0, "A": 0, "B": 0}

    # -------------------------
    # Evaluation Loop
    # -------------------------
    for ex in examples:
        if len(ex) < 4:
            continue

        task_type = ex[1]  # [N] / [A] / [B]

        if task_type == "N":
            # [N]321+654=975
            prefix, answer = ex.split('=')
            prompt = prefix + '='

        elif task_type in ["A", "B"]:
            # [A]X+654=975?321
            prefix, answer = ex.split('?')
            prompt = prefix + '?'

        else:
            continue

        total[task_type] += 1

        # -------------------------
        # Autoregressive Generation
        # -------------------------
        x = torch.tensor(
            [stoi[c] for c in prompt],
            dtype=torch.long,
            device=device
        ).unsqueeze(0)

        generated = ""
        current_x = x

        with torch.no_grad():
            for _ in range(len(answer) + 2):
                if is_loop:
                    logits, _, _ = model(current_x)
                else:
                    logits, _ = model(current_x)

                next_id = torch.argmax(logits[0, -1, :]).item()
                next_char = itos[next_id]

                if next_char == '\n':
                    break

                generated += next_char
                next_tensor = torch.tensor([[next_id]], device=device)
                current_x = torch.cat((current_x, next_tensor), dim=1)

                if current_x.size(1) > gptconf.block_size:
                    current_x = current_x[:, 1:]

        if generated.strip() == answer.strip():
            correct[task_type] += 1

    # -------------------------
    # Report
    # -------------------------
    print("\n" + "=" * 60)
    print("📊 Algebra V2 OOD Intelligence Report")
    print("=" * 60)

    for t in ["N", "A", "B"]:
        if total[t] > 0:
            acc = 100 * correct[t] / total[t]
            print(f"[{t}] Accuracy: {acc:6.2f}% ({correct[t]}/{total[t]})")

    print("=" * 60)


if __name__ == "__main__":
    import sys
    ckpt = sys.argv[1]
    evaluate_algebra_v2(ckpt)