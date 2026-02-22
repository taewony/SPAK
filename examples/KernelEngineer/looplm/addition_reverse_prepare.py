import os
import pickle
import numpy as np
import random

# Configuration
num_samples = 200000 
max_digits_train = 4 
max_digits_ood = 12
script_dir = os.path.dirname(__file__) or '.'
out_dir = os.path.join(script_dir, 'data', 'addition_reverse')
os.makedirs(out_dir, exist_ok=True)

def generate_addition_example_reversed(max_digits):
    d1 = random.randint(1, max_digits)
    d2 = random.randint(1, max_digits)
    n1 = random.randint(0, 10**d1 - 1)
    n2 = random.randint(0, 10**d2 - 1)
    ans = n1 + n2
    # Format: "123+456=975" (where 579 is the answer, but written reversed)
    # This aligns with the model learning to add from right to left!
    return f"{n1}+{n2}={str(ans)[::-1]}"

def generate_addition_data():
    print(f"Generating {num_samples} REVERSED training samples...")
    dataset = []
    for _ in range(num_samples):
        if random.random() < 0.05:
            max_d = random.randint(5, 6)
        else:
            max_d = max_digits_train
        dataset.append(generate_addition_example_reversed(max_d))
    
    print(f"Generating 5000 REVERSED OOD samples...")
    ood_dataset = [generate_addition_example_reversed(max_digits_ood) for _ in range(5000)]

    all_text = "\n".join(dataset) + "\n" + "\n".join(ood_dataset) + "\n"
    chars = sorted(list(set(all_text)))
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    
    train_text = "\n".join(dataset[0:int(len(dataset)*0.9)]) + "\n"
    val_text = "\n".join(dataset[int(len(dataset)*0.9):]) + "\n"
    ood_text = "\n".join(ood_dataset) + "\n"
    
    np.array([stoi[c] for c in train_text], dtype=np.uint16).tofile(os.path.join(out_dir, 'train.bin'))
    np.array([stoi[c] for c in val_text], dtype=np.uint16).tofile(os.path.join(out_dir, 'val.bin'))
    np.array([stoi[c] for c in ood_text], dtype=np.uint16).tofile(os.path.join(out_dir, 'val_ood.bin'))
    
    meta = {'vocab_size': len(chars), 'itos': itos, 'stoi': stoi}
    with open(os.path.join(out_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    print("\n--- Reversed Training Sample ---")
    print(dataset[0]) # e.g., 12+34=64 (instead of 46)

if __name__ == "__main__":
    generate_addition_data()
