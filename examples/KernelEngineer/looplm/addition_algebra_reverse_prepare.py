import os
import pickle
import numpy as np
import random

# -------------------------------------------------
# Configuration
# dataset = 'algebra_reverse'
# weight_decay = 1e-3
# dropout = 0.0
# -------------------------------------------------
num_samples = 300000
max_digits_train = 4
max_digits_ood = 12

script_dir = os.path.dirname(__file__) or '.'
out_dir = os.path.join(script_dir, 'data', 'algebra_reverse')
os.makedirs(out_dir, exist_ok=True)


# -------------------------------------------------
# Carry-heavy number generator
# -------------------------------------------------

def generate_carry_heavy_pair(max_digits):
    d = random.randint(2, max_digits)

    if random.random() < 0.5:
        # Force carry chain
        n1 = int("9" * (d-1) + str(random.randint(0,9)))
        n2 = random.randint(1, 10**d - 1)
    else:
        n1 = random.randint(0, 10**d - 1)
        n2 = random.randint(0, 10**d - 1)

    return n1, n2


# -------------------------------------------------
# Example Generator
# -------------------------------------------------

def generate_example(max_digits):
    n1, n2 = generate_carry_heavy_pair(max_digits)
    ans = n1 + n2

    s1 = str(n1)[::-1]
    s2 = str(n2)[::-1]
    s_ans = str(ans)[::-1]

    task = random.choice(["N", "A", "B"])

    if task == "N":
        # Normal
        return f"[N]{s1}+{s2}={s_ans}"

    elif task == "A":
        # Missing A
        return f"[A]X+{s2}={s_ans}?{s1}"

    elif task == "B":
        # Missing B
        return f"[B]{s1}+X={s_ans}?{s2}"


# -------------------------------------------------
# Data Generation
# -------------------------------------------------

def generate_data():
    print(f"Generating {num_samples} Algebra samples...")
    dataset = []

    for _ in range(num_samples):
        dataset.append(generate_example(max_digits_train))

    print("Generating OOD samples...")
    ood_dataset = [
        generate_example(max_digits_ood)
        for _ in range(5000)
    ]

    # Build vocab
    all_text = "\n".join(dataset) + "\n" + "\n".join(ood_dataset) + "\n"
    chars = sorted(list(set(all_text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    train_text = "\n".join(dataset[: int(len(dataset) * 0.9)]) + "\n"
    val_text = "\n".join(dataset[int(len(dataset) * 0.9):]) + "\n"
    ood_text = "\n".join(ood_dataset) + "\n"

    np.array([stoi[c] for c in train_text], dtype=np.uint16).tofile(
        os.path.join(out_dir, 'train.bin')
    )
    np.array([stoi[c] for c in val_text], dtype=np.uint16).tofile(
        os.path.join(out_dir, 'val.bin')
    )
    np.array([stoi[c] for c in ood_text], dtype=np.uint16).tofile(
        os.path.join(out_dir, 'val_ood.bin')
    )

    meta = {'vocab_size': len(chars), 'itos': itos, 'stoi': stoi}
    with open(os.path.join(out_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    print("\n--- Sample ---")
    for i in range(5):
        print(dataset[i])

    print(f"\nData saved to {out_dir}")


if __name__ == "__main__":
    generate_data()