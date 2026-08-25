"""
data_gen.py
Generates synthetic RNA-like sequences with nested (non-crossing) secondary
structure, similar to how real RNA secondary structure works.

Alphabet: A, U, G, C
Pairing rules (Watson-Crick + wobble, like real RNA):
    A-U, U-A, G-C, C-G, G-U, U-G

Structure is generated with a simple stochastic context-free grammar (SCFG),
which is the standard way to synthesize plausible nested RNA-like structures:

    S -> a S            (unpaired base, continue)
    S -> ( S ) S         (a base pair enclosing a substructure, followed by more)
    S -> ""              (empty / end)

The grammar naturally produces *nested, non-crossing* pairs -- exactly the
kind of long-distance, structurally-nested dependency that a standard
Transformer's local-attention bias struggles with as sequence length grows.

Output format per example:
    sequence: "AUGCUA..."         (raw bases)
    structure: "..((...))."       (dot-bracket notation: '.' unpaired,
                                    '(' opens a pair, ')' closes a pair)
    labels: [0,0,1,1,0,0,0,2,2,0] (0=unpaired, 1=open, 2=close) -- model target
"""

import random
import json
from dataclasses import dataclass, asdict

BASES = ["A", "U", "G", "C"]
PAIR_PARTNERS = {
    "A": ["U"],
    "U": ["A", "G"],  # wobble U-G
    "G": ["C", "U"],  # wobble G-U
    "C": ["G"],
}


@dataclass
class Example:
    sequence: str
    structure: str
    labels: list  # 0 unpaired, 1 open, 2 close
    length: int


def _sample_base() -> str:
    return random.choice(BASES)


def _generate_structure(
    target_len: int,
    p_pair: float = 0.45,
    min_loop: int = 3,
) -> str:
    """
    Recursively build a dot-bracket structure of length <= target_len using
    the SCFG described above. min_loop enforces a minimum hairpin loop size
    (mirrors the real biophysical constraint that RNA can't fold back on
    itself with zero slack), which keeps some pairs local and forces others
    to be genuinely long-range as target_len grows.
    """
    if target_len <= 0:
        return ""

    # Small remaining budget: just emit unpaired bases.
    if target_len < min_loop + 2:
        return "." * target_len

    if random.random() < p_pair:
        # Choose how much of the remaining length goes *inside* this pair.
        inside_budget = random.randint(min_loop, target_len - 2)
        inside = _generate_structure(inside_budget, p_pair, min_loop)
        remaining = target_len - 2 - len(inside)
        after = _generate_structure(remaining, p_pair, min_loop)
        return "(" + inside + ")" + after
    else:
        after = _generate_structure(target_len - 1, p_pair, min_loop)
        return "." + after


def _fill_sequence_from_structure(structure: str) -> str:
    """
    Walk the dot-bracket structure and assign bases such that every
    matched pair is chemically valid (A-U, G-C, G-U, etc.), and unpaired
    positions get a random base. This mimics real RNA sequence/structure
    covariation.
    """
    seq = [None] * len(structure)
    stack = []
    for i, c in enumerate(structure):
        if c == ".":
            seq[i] = _sample_base()
        elif c == "(":
            b = _sample_base()
            seq[i] = b
            stack.append((i, b))
        elif c == ")":
            j, b_open = stack.pop()
            seq[i] = random.choice(PAIR_PARTNERS[b_open])
    return "".join(seq)


def generate_example(length: int, p_pair: float = 0.45, min_loop: int = 3) -> Example:
    structure = _generate_structure(length, p_pair=p_pair, min_loop=min_loop)
    # padding safety: grammar can occasionally return slightly shorter string
    if len(structure) < length:
        structure += "." * (length - len(structure))
    sequence = _fill_sequence_from_structure(structure)
    label_map = {".": 0, "(": 1, ")": 2}
    labels = [label_map[c] for c in structure]
    return Example(sequence=sequence, structure=structure, labels=labels, length=length)


def generate_dataset(
    n_examples: int,
    length_range: tuple,
    p_pair: float = 0.45,
    min_loop: int = 3,
    seed: int = 0,
):
    random.seed(seed)
    examples = []
    lo, hi = length_range
    for _ in range(n_examples):
        length = random.randint(lo, hi)
        examples.append(generate_example(length, p_pair=p_pair, min_loop=min_loop))
    return examples


def save_jsonl(examples, path):
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex)) + "\n")


if __name__ == "__main__":
    import os

    os.makedirs("data", exist_ok=True)

    # Train on a mix of short-to-medium sequences.
    train_examples = generate_dataset(20000, length_range=(20, 200), seed=1)
    val_examples = generate_dataset(2000, length_range=(20, 200), seed=2)

    # Held-out length bins to measure how performance degrades with distance
    # -- this is the actual point of the experiment.
    test_bins = {
        "test_short_20_60": generate_dataset(1000, length_range=(20, 60), seed=10),
        "test_medium_60_150": generate_dataset(1000, length_range=(60, 150), seed=11),
        "test_long_150_300": generate_dataset(1000, length_range=(150, 300), seed=12),
        "test_verylong_300_500": generate_dataset(1000, length_range=(300, 500), seed=13),
    }

    save_jsonl(train_examples, "data/train.jsonl")
    save_jsonl(val_examples, "data/val.jsonl")
    for name, exs in test_bins.items():
        save_jsonl(exs, f"data/{name}.jsonl")

    print(f"train: {len(train_examples)} examples")
    print(f"val:   {len(val_examples)} examples")
    for name, exs in test_bins.items():
        print(f"{name}: {len(exs)} examples")

    print("\nsample:")
    ex = train_examples[0]
    print("seq :", ex.sequence[:60])
    print("struct:", ex.structure[:60])
