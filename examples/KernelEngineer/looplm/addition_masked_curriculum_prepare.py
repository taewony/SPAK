import os
import pickle
import numpy as np
import random

# -------------------------------------------------
# Configuration
# -------------------------------------------------
num_samples = 200000

max_digits_train = 4
max_digits_bridge_low = 5
max_digits_bridge_high = 6
max_digits_ood = 12

mask_token = 'X'

script_dir = os.path.dirname(__file__) or '.'
out_dir = os.path.join(script_dir, 'data', 'addition_masked_curriculum')
os.makedirs(out_dir, exist_ok=True)


# -------------------------------------------------
# Masking Utility
# -------------------------------------------------

def mask_digits(s, mask_prob):
    s = list(s)
    masked_positions = []

    for i in range(len(s)):
        if s[i].isdigit() and random.random() < mask_prob:
            s[i] = mask_token
            masked_positions.append(i)

    return "".join(s), masked_positions


def force_at_least_one_mask(s, mask_prob):
    s_masked, positions = mask_digits(s, mask_prob)

    if len(positions) == 0:
        digit_indices = [i for i, c in enumerate(s) if c.isdigit()]
        if digit_indices:
            idx = random.choice(digit_indices)
            s_masked = list(s)
            s_masked[idx] = mask_token
            s_masked = "".join(s_masked)

    return s_masked


# -------------------------------------------------
# Example Generator
# -------------------------------------------------

def generate_masked_addition_example(max_digits, mask_prob_input, mask_prob_output):
    d1 = random.randint(1, max_digits)
    d2 = random.randint(1, max_digits)

    n1 = random.randint(0, 10**d1 - 1)
    n2 = random.randint(0, 10**d2 - 1)
    ans = n1 + n2

    # reverse formatting
    s1_rev = str(n1)[::-1]
    s2_rev = str(n2)[::-1]
    ans_rev = str(ans)[::-1]

    # masking
    s1_masked = force_at_least_one_mask(s1_rev, mask_prob_input)
    s2_masked = force_at_least_one_mask(s2_rev, mask_prob_input)
    ans_masked = force_at_least_one_mask(ans_rev, mask_prob_output)

    return f"{s1_masked}+{s2_masked}={ans_masked}"


# -------------------------------------------------
# Curriculum Stage Selector
# -------------------------------------------------

def get_curriculum_stage(sample_index):
    ratio = sample_index / num_samples

    if ratio < 0.33:
        # Stage 1
        return 0.05, 0.10
    elif ratio < 0.66:
        # Stage 2
        return 0.10, 0.20
    else:
        # Stage 3
        return 0.15, 0.35


# -------------------------------------------------
# Data Generation
# -------------------------------------------------

def generate_addition_data():

    print(f"Generating {num_samples} CURRICULUM masked training samples...")

    dataset = []

    for i in range(num_samples):

        # digit range curriculum 유지
        if random.random() < 0.30:
            max_d = random.randint(max_digits_bridge_low, max_digits_bridge_high)
        else:
            max_d = max_digits_train

        mask_prob_input, mask_prob_output = get_curriculum_stage(i)

        example = generate_masked_addition_example(
            max_d,
            mask_prob_input,
            mask_prob_output
        )

        dataset.append(example)

    print("Generating OOD masked samples...")

    ood_dataset = [
        generate_masked_addition_example(
            max_digits_ood,
            0.20,
            0.40
        )
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

    print("\n--- Sample (Easy) ---")
    print(dataset[0])

    print("\n--- Sample (Mid) ---")
    print(dataset[int(num_samples * 0.5)])

    print("\n--- Sample (Hard) ---")
    print(dataset[-1])


if __name__ == "__main__":
    generate_addition_data()