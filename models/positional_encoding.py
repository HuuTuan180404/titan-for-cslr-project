import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int, dropout: float = 0.0):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        # Shape: (max_seq_len, d_model)
        pe = torch.zeros(max_seq_len, d_model)

        # Shape: (max_seq_len, 1)
        position = torch.arange(max_seq_len).unsqueeze(1)

        # Shape: (d_model // 2,)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        # Even dimensions: sin
        pe[:, 0::2] = torch.sin(position * div_term)

        # Odd dimensions: cos
        pe[:, 1::2] = torch.cos(position * div_term)

        # Shape: (1, max_seq_len, d_model)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x + self.pe[:, :x.size(1), :]

        return self.dropout(x)