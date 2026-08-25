"""
pair_aux.py
Auxiliary pair-matching loss (Experiment 2).

Token-level cross-entropy (even weighted) only supervises "what is the
label at position i" -- it never tells the model "position i's matching
partner is position j". For a structure-recovery task where the entire
point is nested long-range matching, that's a weak training signal: the
model can get partial credit for correctly marking a position as '(' or
')' without ever getting gradient information about the *specific* partner
position, i.e. exactly the distance-|i-j| information from the M.Sc.
Fredholm-kernel background this project is built around.

PairHead computes a pairwise compatibility score matrix over the sequence
(a pointer-network-style mechanism: for each position, "which other
position is my partner?"), and pair_loss supervises it directly against
the ground-truth partner index for every true open/close position --
completely separate from (and complementary to) the per-position token
classification loss.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PairHead(nn.Module):
    def __init__(self, d_model: int, proj_dim: int = 64):
        super().__init__()
        self.proj_dim = proj_dim
        self.query_proj = nn.Linear(d_model, proj_dim)
        self.key_proj = nn.Linear(d_model, proj_dim)

    def forward(self, hidden, attention_mask=None):
        # hidden: (batch, seq_len, d_model)
        q = self.query_proj(hidden)
        k = self.key_proj(hidden)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.proj_dim)
        # scores: (batch, seq_len, seq_len) -- scores[b, i, j] = compatibility
        # of position i with position j as its pairing partner

        if attention_mask is not None:
            pad_mask = attention_mask[:, None, :] == 0  # (batch, 1, seq_len), True = pad column
            scores = scores.masked_fill(pad_mask, float("-inf"))

        return scores


def build_partner_targets(labels, ignore_index=-100):
    """
    For each sequence in the batch, walks the true dot-bracket labels
    (0=unpaired, 1=open, 2=close) with a stack to find every true pair
    (i, j), and builds a (batch, seq_len) target tensor where:
      - target[b, i] = j   if position i is a true open paren
      - target[b, j] = i   if position j is a true close paren
      - target[b, k] = ignore_index otherwise (unpaired, or padding)
    This only uses ground-truth labels (available during training), never
    model predictions, so it's a valid supervised target.
    """
    batch_size, seq_len = labels.shape
    targets = torch.full_like(labels, ignore_index)

    labels_cpu = labels.cpu()
    for b in range(batch_size):
        stack = []
        for i, lab in enumerate(labels_cpu[b].tolist()):
            if lab == 1:
                stack.append(i)
            elif lab == 2 and stack:
                j = stack.pop()
                targets[b, j] = i  # open position j's partner is close position i
                targets[b, i] = j  # close position i's partner is open position j
    return targets


def pair_loss(scores, labels, ignore_index=-100):
    """
    scores: (batch, seq_len, seq_len) from PairHead
    labels: (batch, seq_len) true per-position labels (0/1/2, -100 for pad)
    """
    targets = build_partner_targets(labels, ignore_index=ignore_index).to(scores.device)
    batch, seq_len, _ = scores.shape
    loss = F.cross_entropy(
        scores.reshape(batch * seq_len, seq_len),
        targets.reshape(batch * seq_len),
        ignore_index=ignore_index,
    )
    return loss


def pair_head_accuracy(scores, labels, ignore_index=-100):
    """
    Diagnostic metric: of all true open/close positions, what fraction does
    the pair head's argmax correctly point to its true partner? This is a
    more direct/optimistic signal than the end-to-end pair_match_accuracy
    (which requires the token classifier to also get both endpoints' labels
    right), useful for checking whether the pairing structure itself is
    being learned even before the classifier head catches up.
    """
    targets = build_partner_targets(labels, ignore_index=ignore_index).to(scores.device)
    preds = scores.argmax(dim=-1)
    mask = targets != ignore_index
    if mask.sum() == 0:
        return 0.0
    correct = (preds[mask] == targets[mask]).float().mean().item()
    return correct
