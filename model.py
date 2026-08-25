"""
model.py
Baseline: a standard Transformer encoder (PyTorch nn.TransformerEncoder,
sinusoidal positional encoding, vanilla dot-product self-attention) doing
per-position 3-way classification: unpaired / open-pair / close-pair.

This is the reference point the interpretability / long-range experiments
will be compared against. Nothing exotic here on purpose -- the whole point
of the project is to first establish *how badly* standard attention degrades
as the pairing distance grows, before trying a kernel/operator-inspired
alternative.
"""

import math
import torch
import torch.nn as nn

VOCAB = {"A": 0, "U": 1, "G": 2, "C": 3, "<pad>": 4}
NUM_CLASSES = 3  # unpaired, open, close


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, : x.size(1)]


class BaselineTransformer(nn.Module):
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
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, input_ids, attention_mask=None, return_hidden=False):
        # input_ids: (batch, seq_len)
        # attention_mask: (batch, seq_len), 1 = real token, 0 = pad
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = attention_mask == 0  # True where padded

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        logits = self.classifier(x)  # (batch, seq_len, num_classes)
        if return_hidden:
            return logits, x
        return logits


def encode_sequence(seq: str, max_len: int = None):
    ids = [VOCAB[c] for c in seq]
    if max_len is not None:
        ids = ids + [VOCAB["<pad>"]] * (max_len - len(ids))
    return ids
