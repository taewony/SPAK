import os
import pickle
import numpy as np
import random

# Configuration
num_samples = 50000
max_digits = 4 # Training on up to 4 digits
out_dir = 'looplm/data/addition'
os.makedirs(out_dir, exist_ok=True)

def generate_addition_data():
    dataset = []
    for _ in range(num_samples):
        # Sample random digits
        d1 = random.randint(1, max_digits)
        d2 = random.randint(1, max_digits)
        n1 = random.randint(0, 10**d1 - 1)
        n2 = random.randint(0, 10**d2 - 1)
        
        # Format: "12+34=46
"
        example = f"{n1}+{n2}={n1+n2}
"
        dataset.append(example)
    
    # Shuffle
    random.shuffle(dataset)
    text = "".join(dataset)
    
    # Vocabulary (Characters: 0-9, +, =, 
)
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    
    print(f"Dataset size: {len(text)} chars")
    print(f"Vocab size: {vocab_size}, chars: {''.join(chars)}")
    
    # Split
    n = len(text)
    train_data = text[:int(n*0.9)]
    val_data = text[int(n*0.9):]
    
    # Encode
    train_ids = np.array([stoi[c] for c in train_data], dtype=np.uint16)
    val_ids = np.array([stoi[c] for c in val_data], dtype=np.uint16)
    
    # Save
    train_ids.tofile(os.path.join(out_dir, 'train.bin'))
    val_ids.tofile(os.path.join(out_dir, 'val.bin'))
    
    meta = {'vocab_size': vocab_size, 'itos': itos, 'stoi': stoi}
    with open(os.path.join(out_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

if __name__ == "__main__":
    generate_addition_data()
