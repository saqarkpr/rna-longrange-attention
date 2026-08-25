"""
train.py
Trains the baseline Transformer on the synthetic RNA-like pairing task and
evaluates it separately on each held-out length bin. The key output is a
table/plot of accuracy vs. sequence length -- this is the evidence for
"standard attention degrades on long-range structural dependencies", which
is the whole premise of the follow-up (kernel/operator-attention) experiment.

Usage:
    python data_gen.py     # generate data/*.jsonl once
    python train.py
"""

import json
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import BaselineTransformer, VOCAB, encode_sequence

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PairingDataset(Dataset):
    def __init__(self, path):
        self.examples = []
        with open(path) as f:
            for line in f:
                self.examples.append(json.loads(line))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return ex["sequence"], ex["labels"]


def collate_fn(batch):
    seqs, labels = zip(*batch)
    max_len = max(len(s) for s in seqs)
    input_ids = torch.tensor([encode_sequence(s, max_len) for s in seqs], dtype=torch.long)
    attention_mask = torch.zeros(len(seqs), max_len, dtype=torch.long)
    label_tensor = torch.full((len(seqs), max_len), fill_value=-100, dtype=torch.long)
    for i, (s, l) in enumerate(zip(seqs, labels)):
        attention_mask[i, : len(s)] = 1
        label_tensor[i, : len(l)] = torch.tensor(l, dtype=torch.long)
    return input_ids, attention_mask, label_tensor


def run_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, total_correct, total_tokens = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    with torch.set_grad_enabled(is_train):
        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            preds = logits.argmax(dim=-1)
            mask = labels != -100
            total_correct += (preds[mask] == labels[mask]).sum().item()
            total_tokens += mask.sum().item()
            total_loss += loss.item() * mask.sum().item()

    return total_loss / total_tokens, total_correct / total_tokens


def pairing_accuracy(model, loader):
    """
    Stricter metric than raw token accuracy: for every true '(' at position i
    matched with ')' at position j, check whether the model's argmax
    predictions also mark i as open and j as close AND get the *matching*
    right (not just the open/close label). This is what actually matters for
    "did the model recover the long-range dependency", vs. token accuracy
    which is dominated by the easy unpaired class.
    """
    model.eval()
    correct_pairs, total_pairs = 0, 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            logits = model(input_ids, attention_mask)
            preds = logits.argmax(dim=-1).cpu()
            labels_cpu = labels.cpu()

            for b in range(input_ids.size(0)):
                true_seq = labels_cpu[b]
                pred_seq = preds[b]
                length = (attention_mask[b].cpu() == 1).sum().item()
                true_seq = true_seq[:length]
                pred_seq = pred_seq[:length]

                # reconstruct true pairs via stack
                stack = []
                true_pairs = []
                for i, lab in enumerate(true_seq.tolist()):
                    if lab == 1:
                        stack.append(i)
                    elif lab == 2 and stack:
                        j = stack.pop()
                        true_pairs.append((j, i))

                pred_labels = pred_seq.tolist()
                for (i, j) in true_pairs:
                    total_pairs += 1
                    if pred_labels[i] == 1 and pred_labels[j] == 2:
                        correct_pairs += 1

    return correct_pairs / max(total_pairs, 1)


def main():
    train_ds = PairingDataset("data/train.jsonl")
    val_ds = PairingDataset("data/val.jsonl")

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    model = BaselineTransformer().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    n_epochs = 8
    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader)
        print(
            f"epoch {epoch}/{n_epochs}  "
            f"train_loss={train_loss:.4f} train_tok_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f} val_tok_acc={val_acc:.4f}  "
            f"({time.time()-t0:.1f}s)"
        )

    torch.save(model.state_dict(), "baseline_transformer.pt")
    print("saved model to baseline_transformer.pt")

    # Evaluate by length bin -- this is the actual result we care about.
    print("\n--- accuracy by sequence length bin ---")
    bins = [
        "test_short_20_60",
        "test_medium_60_150",
        "test_long_150_300",
        "test_verylong_300_500",
    ]
    results = {}
    for name in bins:
        ds = PairingDataset(f"data/{name}.jsonl")
        loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_fn)
        _, tok_acc = run_epoch(model, loader)
        pair_acc = pairing_accuracy(model, loader)
        results[name] = {"token_accuracy": tok_acc, "pair_match_accuracy": pair_acc}
        print(f"{name:25s}  token_acc={tok_acc:.4f}  pair_match_acc={pair_acc:.4f}")

    with open("baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
