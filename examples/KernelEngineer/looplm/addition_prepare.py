import os
import pickle
import numpy as np
import random

# Configuration
num_samples = 200000 # Increased for Phase 3
max_digits_train = 4 
max_digits_ood = 12
# Get absolute path to the directory where this script is located
script_dir = os.path.dirname(__file__)
if script_dir == '': script_dir = '.'
out_dir = os.path.join(script_dir, 'data', 'addition')
os.makedirs(out_dir, exist_ok=True)

def generate_addition_example(max_digits):
    # Sample random digits
    d1 = random.randint(1, max_digits)
    d2 = random.randint(1, max_digits)
    n1 = random.randint(0, 10**d1 - 1)
    n2 = random.randint(0, 10**d2 - 1)
    return f"{n1}+{n2}={n1+n2}"

def generate_addition_data():
    print(f"Generating {num_samples} training samples (with 5% Bridge Data)...")
    dataset = []
    for _ in range(num_samples):
        # 95% 1-4 digits, 5% 5-6 digits (The Bridge)
        if random.random() < 0.05:
            max_d = random.randint(5, 6)
        else:
            max_d = max_digits_train
        dataset.append(generate_addition_example(max_d))
    
    print(f"Generating 5000 OOD samples (up to {max_digits_ood} digits)...")
    ood_dataset = [generate_addition_example(max_digits_ood) for _ in range(5000)]

    # Shuffle
    random.shuffle(dataset)
    
    # Vocabulary (Characters: 0-9, +, =, \n)
    all_text = "\n".join(dataset) + "\n" + "\n".join(ood_dataset) + "\n"
    chars = sorted(list(set(all_text)))
    vocab_size = len(chars)
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    
    print(f"Dataset size: {len(all_text)} chars")
    print(f"Vocab size: {vocab_size}, chars: {repr(''.join(chars))}")
    
    # Split training/val
    train_text = "\n".join(dataset[:int(len(dataset)*0.9)]) + "\n"
    val_text = "\n".join(dataset[int(len(dataset)*0.9):]) + "\n"
    ood_text = "\n".join(ood_dataset) + "\n"
    
    # Encode
    train_ids = np.array([stoi[c] for c in train_text], dtype=np.uint16)
    val_ids = np.array([stoi[c] for c in val_text], dtype=np.uint16)
    ood_ids = np.array([stoi[c] for c in ood_text], dtype=np.uint16)
    
    # Save
    train_ids.tofile(os.path.join(out_dir, 'train.bin'))
    val_ids.tofile(os.path.join(out_dir, 'val.bin'))
    ood_ids.tofile(os.path.join(out_dir, 'val_ood.bin')) # Added OOD
    
    meta = {'vocab_size': vocab_size, 'itos': itos, 'stoi': stoi}
    with open(os.path.join(out_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    print("\n--- Training Samples ---")
    for i in range(3): print(dataset[i])
    print("\n--- OOD Samples ---")
    for i in range(3): print(ood_dataset[i])

if __name__ == "__main__":
    generate_addition_data()
