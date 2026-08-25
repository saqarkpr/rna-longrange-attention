"""
train_utils.py
Model-agnostic training / evaluation helpers, factored out of train.py so
both BaselineTransformer and KernelAttentionTransformer can reuse the exact
same data loading, training loop, and evaluation metrics -- necessary for a
fair, controlled comparison (same data, same optimizer, same epochs, same
metrics).
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import VOCAB, encode_sequence
from pair_aux import PairHead, pair_loss, pair_head_accuracy

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


def make_loader(path, batch_size=64, shuffle=False):
    ds = PairingDataset(path)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def compute_class_weights(loader, num_classes=3):
    """
    Computes inverse-frequency class weights from a data loader. The
    pairing task has a strong class imbalance (~65% unpaired, ~17.5% open,
    ~17.5% close); plain cross-entropy lets the model settle into a local
    minimum of "always predict unpaired" because that's a very safe,
    low-loss solution given the imbalance, and the gradient signal for
    learning the actual long-range pairing is comparatively weak. Weighting
    the loss by inverse class frequency counteracts this.
    """
    counts = torch.zeros(num_classes)
    for _, _, labels in loader:
        for c in range(num_classes):
            counts[c] += (labels == c).sum()
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes  # normalize, keep scale reasonable
    return weights


def run_epoch(model, loader, optimizer=None, scheduler=None, class_weights=None,
              pair_head=None, pair_loss_weight=1.0, pair_optimizer=None):
    """
    pair_head: optional PairHead module. When provided, the token-level
    cross-entropy loss is combined with the pair-matching auxiliary loss:
        L = L_token + pair_loss_weight * L_pair
    pair_optimizer: if pair_head has its own parameters (it does), pass an
    optimizer covering them too (or include pair_head.parameters() in the
    main optimizer -- either works, see train_model).
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    if pair_head is not None:
        pair_head.train() if is_train else pair_head.eval()

    total_loss, total_correct, total_tokens = 0.0, 0, 0
    total_pair_loss, total_pair_head_acc, n_batches = 0.0, 0.0, 0
    cw = class_weights.to(DEVICE) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(ignore_index=-100, weight=cw)

    with torch.set_grad_enabled(is_train):
        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            labels = labels.to(DEVICE)

            if pair_head is not None:
                logits, hidden = model(input_ids, attention_mask, return_hidden=True)
                scores = pair_head(hidden, attention_mask)
                p_loss = pair_loss(scores, labels)
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1)) + pair_loss_weight * p_loss
                total_pair_loss += p_loss.item()
                total_pair_head_acc += pair_head_accuracy(scores, labels)
                n_batches += 1
            else:
                logits = model(input_ids, attention_mask)
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            if is_train:
                optimizer.zero_grad()
                if pair_optimizer is not None:
                    pair_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if pair_optimizer is not None:
                    pair_optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            preds = logits.argmax(dim=-1)
            mask = labels != -100
            total_correct += (preds[mask] == labels[mask]).sum().item()
            total_tokens += mask.sum().item()
            total_loss += loss.item() * mask.sum().item()

    tok_loss = total_loss / total_tokens
    tok_acc = total_correct / total_tokens
    if pair_head is not None:
        return tok_loss, tok_acc, total_pair_head_acc / max(n_batches, 1)
    return tok_loss, tok_acc


def pairing_accuracy(model, loader):
    """
    For every true base pair (i, j), checks whether the model predicts BOTH
    i as open AND j as close. This is the metric that actually reflects
    long-range structural recovery (token accuracy is dominated by the easy
    "unpaired" class).
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
                length = (attention_mask[b].cpu() == 1).sum().item()
                true_seq = labels_cpu[b][:length]
                pred_seq = preds[b][:length]

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


def train_model(model, train_loader, val_loader, n_epochs=8, lr=3e-4, tag="model", warmup_steps=500,
                 use_class_weights=True, use_pair_aux=False, pair_loss_weight=1.0, pair_proj_dim=64):
    """
    warmup_steps: linear LR warmup from 0 to `lr` over this many optimizer
    steps, then constant.

    use_class_weights: the pairing task is class-imbalanced (~65% unpaired,
    ~17.5% open, ~17.5% close). A full-scale run (20k examples, 30 epochs,
    unweighted loss) showed BOTH the baseline and kernel-attention models
    getting stuck exactly at the majority-class base rate (~64.6% token
    accuracy, 0% pair-match accuracy) for the entire run -- i.e. "always
    predict unpaired" is a very safe local minimum that plain cross-entropy
    does not push the model out of. Weighting the loss by inverse class
    frequency counteracts this (Experiment 1).

    use_pair_aux: adds a pointer-network-style PairHead trained with a
    direct pair-matching loss (Experiment 2). Token classification alone
    only supervises "what label is at position i"; it never directly
    supervises "position i's partner is position j". The pair-aux loss
    does exactly that, using ground-truth partner indices derived from the
    true labels, and is combined with the token loss as
        L = L_token + pair_loss_weight * L_pair
    """
    model = model.to(DEVICE)

    pair_head = None
    if use_pair_aux:
        d_model = model.d_model
        pair_head = PairHead(d_model, proj_dim=pair_proj_dim).to(DEVICE)
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(pair_head.parameters()), lr=lr
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    class_weights = compute_class_weights(train_loader) if use_class_weights else None
    if class_weights is not None:
        print(f"[{tag}] class weights (unpaired, open, close): {class_weights.tolist()}")

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    for epoch in range(1, n_epochs + 1):
        if use_pair_aux:
            train_loss, train_acc, train_pair_head_acc = run_epoch(
                model, train_loader, optimizer, scheduler=scheduler, class_weights=class_weights,
                pair_head=pair_head, pair_loss_weight=pair_loss_weight,
            )
            val_loss, val_acc, val_pair_head_acc = run_epoch(
                model, val_loader, class_weights=class_weights,
                pair_head=pair_head, pair_loss_weight=pair_loss_weight,
            )
            print(
                f"[{tag}] epoch {epoch}/{n_epochs}  "
                f"train_loss={train_loss:.4f} train_tok_acc={train_acc:.4f} train_pairhead_acc={train_pair_head_acc:.4f}  "
                f"val_loss={val_loss:.4f} val_tok_acc={val_acc:.4f} val_pairhead_acc={val_pair_head_acc:.4f}"
            )
        else:
            train_loss, train_acc = run_epoch(model, train_loader, optimizer, scheduler=scheduler, class_weights=class_weights)
            val_loss, val_acc = run_epoch(model, val_loader, class_weights=class_weights)
            print(
                f"[{tag}] epoch {epoch}/{n_epochs}  "
                f"train_loss={train_loss:.4f} train_tok_acc={train_acc:.4f}  "
                f"val_loss={val_loss:.4f} val_tok_acc={val_acc:.4f}"
            )
    return model, pair_head


def evaluate_by_length_bin(model, bin_paths: dict, tag="model"):
    results = {}
    print(f"\n--- [{tag}] accuracy by sequence length bin ---")
    for name, path in bin_paths.items():
        loader = make_loader(path, batch_size=64, shuffle=False)
        _, tok_acc = run_epoch(model, loader)
        pair_acc = pairing_accuracy(model, loader)
        results[name] = {"token_accuracy": tok_acc, "pair_match_accuracy": pair_acc}
        print(f"{name:25s}  token_acc={tok_acc:.4f}  pair_match_acc={pair_acc:.4f}")
    return results
