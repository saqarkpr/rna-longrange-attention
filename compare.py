"""
compare.py
Trains BaselineTransformer and KernelAttentionTransformer under identical
conditions (same data, optimizer, epochs, model size) and compares
pair-match accuracy across the four held-out length bins. This is the core
experiment / result of the project.

Usage:
    python data_gen.py
    python compare.py
"""

import json
import torch

from model import BaselineTransformer
from model_kernel import KernelAttentionTransformer
from train_utils import make_loader, train_model, evaluate_by_length_bin

TEST_BINS = {
    "test_short_20_60": "data/test_short_20_60.jsonl",
    "test_medium_60_150": "data/test_medium_60_150.jsonl",
    "test_long_150_300": "data/test_long_150_300.jsonl",
    "test_verylong_300_500": "data/test_verylong_300_500.jsonl",
}

# Keep both models the same size for a fair comparison.
MODEL_KWARGS = dict(d_model=128, nhead=8, num_layers=4, dim_feedforward=256, dropout=0.1)
# NOTE: this task (recovering nested long-range pairing structure) is much
# slower to learn than typical classification -- a debugging run showed the
# model is stuck predicting the majority class for the first ~1500-2000
# optimizer steps before pair-match accuracy starts climbing above zero. 8
# epochs (~2500 steps on the full 20k-example dataset) was not enough,
# hence the increase to 30 epochs and an LR warmup below.
N_EPOCHS = 30
WARMUP_STEPS = 1000
# Experiment 2: pair-aware auxiliary loss (see pair_aux.py). Combined with
# the class-weighted token loss from Experiment 1.
USE_PAIR_AUX = True
# NOTE: a medium-scale (1500 examples, 15 epoch) sanity check found
# pair_loss_weight=1.0 let the (much larger-magnitude) pair-matching loss
# dominate the combined loss and gave only a marginal improvement over
# weighted-CE alone (0.268 vs 0.261 pair-match accuracy); weight=0.3 did
# somewhat better (0.303). This is a single-seed, small-scale result, not
# a confirmed optimum -- worth re-checking once the full-scale run
# completes, and potentially worth tuning further.
PAIR_LOSS_WEIGHT = 0.3


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    train_loader = make_loader("data/train.jsonl", batch_size=64, shuffle=True)
    val_loader = make_loader("data/val.jsonl", batch_size=64, shuffle=False)

    baseline = BaselineTransformer(**MODEL_KWARGS)
    print(f"baseline params: {count_params(baseline):,}")
    baseline, baseline_pair_head = train_model(
        baseline, train_loader, val_loader, n_epochs=N_EPOCHS, tag="baseline", warmup_steps=WARMUP_STEPS,
        use_pair_aux=USE_PAIR_AUX, pair_loss_weight=PAIR_LOSS_WEIGHT,
    )
    torch.save(baseline.state_dict(), "baseline_transformer.pt")
    baseline_results = evaluate_by_length_bin(baseline, TEST_BINS, tag="baseline")
    with open("baseline_results_partial.json", "w") as f:
        json.dump(baseline_results, f, indent=2)

    kernel_model = KernelAttentionTransformer(**MODEL_KWARGS)
    print(f"kernel-attention params: {count_params(kernel_model):,}")
    kernel_model, kernel_pair_head = train_model(
        kernel_model, train_loader, val_loader, n_epochs=N_EPOCHS, tag="kernel", warmup_steps=WARMUP_STEPS,
        use_pair_aux=USE_PAIR_AUX, pair_loss_weight=PAIR_LOSS_WEIGHT,
    )
    torch.save(kernel_model.state_dict(), "kernel_transformer.pt")
    kernel_results = evaluate_by_length_bin(kernel_model, TEST_BINS, tag="kernel")

    comparison = {"baseline": baseline_results, "kernel_attention": kernel_results}
    with open("comparison_results.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print("\n=== SUMMARY: pair-match accuracy by length bin ===")
    print(f"{'bin':25s} {'baseline':>10s} {'kernel':>10s} {'delta':>8s}")
    for name in TEST_BINS:
        b = baseline_results[name]["pair_match_accuracy"]
        k = kernel_results[name]["pair_match_accuracy"]
        print(f"{name:25s} {b:10.4f} {k:10.4f} {k - b:+8.4f}")


if __name__ == "__main__":
    main()
