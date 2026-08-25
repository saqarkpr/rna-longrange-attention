"""
model_kernel.py
Kernel-biased attention: a Transformer variant where self-attention logits
get an additive bias term k(i, j) learned as a function of the *relative
distance* between positions i and j, via a small MLP.

Motivation / connection to prior work (Fredholm integral equations):
Standard numerical solvers for Fredholm integral equations of the form
    f(x) = g(x) + lambda * integral( k(x, y) * f(y) dy )
discretize the unknown *kernel function* k(x, y) on a quadrature grid and
solve the resulting linear system (the Nystrom method). Nystromformer
(Xiong et al., 2021) borrows this exact numerical idea to approximate
softmax attention efficiently.

This module goes the other direction: instead of approximating standard
attention, it *augments* it with an explicitly learned kernel-of-distance
term, so long-range pairs (which are common in RNA secondary structure)
have a dedicated, directly-trainable pathway for "how much should position
i attend to a position |i-j| away", separate from content-based QK
similarity, which is what content attention on its own must otherwise learn
indirectly through many layers.

This is intentionally a *content bias* (relative-position-style, like T5 /
ALiBi) but with a learned nonlinear kernel function instead of a fixed
linear/log decay -- the flexibility is the point, since the pairing
structure here is not "closer = more likely", it is "structured and
long-range", which a monotonic distance penalty (ALiBi) cannot represent
well.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import SinusoidalPositionalEncoding, VOCAB, NUM_CLASSES


class DistanceKernel(nn.Module):
    """
    Learns k(delta) for delta = i - j, one kernel per attention head.
    Implemented as a small MLP over a scalar (signed, log-scaled distance)
    input, shared across all layers that use this instance -- but we
    instantiate one per layer, so each layer can learn a different kernel
    (e.g. a shallow layer might learn a short-range kernel, a deep layer a
    long-range one).
    """

    def __init__(self, nhead: int, hidden: int = 32, max_len: int = 2000):
        super().__init__()
        self.nhead = nhead
        self.max_len = max_len
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, nhead),
        )
        self._cache = {}

    def _signed_log_distance(self, seq_len: int, device):
        idx = torch.arange(seq_len, device=device)
        delta = idx[None, :] - idx[:, None]  # (seq_len, seq_len), i - j
        sign = torch.sign(delta).float()
        mag = torch.log1p(delta.abs().float())
        feats = torch.stack([sign, mag], dim=-1)  # (seq_len, seq_len, 2)
        return feats

    def forward(self, seq_len: int, device):
        key = (seq_len, device)
        if key not in self._cache:
            feats = self._signed_log_distance(seq_len, device)
            self._cache[key] = feats
        feats = self._cache[key]
        bias = self.mlp(feats)  # (seq_len, seq_len, nhead)
        return bias.permute(2, 0, 1)  # (nhead, seq_len, seq_len)


class KernelMultiheadAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1, max_len: int = 2000):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.kernel = DistanceKernel(nhead, max_len=max_len)

    def forward(self, x, key_padding_mask=None):
        # x: (batch, seq_len, d_model)
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        # q,k,v: (batch, nhead, seq_len, head_dim)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # scores: (batch, nhead, seq_len, seq_len)

        kernel_bias = self.kernel(seq_len, x.device)  # (nhead, seq_len, seq_len)
        scores = scores + kernel_bias.unsqueeze(0)  # broadcast over batch

        if key_padding_mask is not None:
            # key_padding_mask: (batch, seq_len), True = pad
            mask = key_padding_mask[:, None, None, :]  # (batch,1,1,seq_len)
            scores = scores.masked_fill(mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (batch, nhead, seq_len, head_dim)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.out_proj(out)


class KernelEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.1, max_len=2000):
        super().__init__()
        self.self_attn = KernelMultiheadAttention(d_model, nhead, dropout=dropout, max_len=max_len)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x, key_padding_mask=None):
        attn_out = self.self_attn(x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout1(attn_out))

        ff_out = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm2(x + self.dropout2(ff_out))
        return x


class KernelAttentionTransformer(nn.Module):
    """
    Same embedding / positional-encoding / classifier head as
    BaselineTransformer, so the two are a controlled comparison -- the only
    difference is the attention mechanism itself.
    """

    def __init__(
        self,
        vocab_size: int = len(VOCAB),
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 2000,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=VOCAB["<pad>"])
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        self.layers = nn.ModuleList(
            [
                KernelEncoderLayer(d_model, nhead, dim_feedforward, dropout, max_len)
                for _ in range(num_layers)
            ]
        )
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, input_ids, attention_mask=None, return_hidden=False):
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0  # True where padded

        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)

        logits = self.classifier(x)
        if return_hidden:
            return logits, x
        return logits
