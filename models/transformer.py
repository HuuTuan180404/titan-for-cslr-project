import torch.nn as nn

from .mlp import MLP


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, act, mlp_ratio, dropout, pre_norm):
        super().__init__()

        self.pre_norm = pre_norm

        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.mlp = MLP(in_dim=d_model, mlp_ratio=mlp_ratio, act=act, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.pre_norm:
            # Attention
            residual = x
            x_norm = self.norm1(x)

            attn_output, attn_weights = self.attn(x_norm, x_norm, x_norm)

            x = residual + self.dropout(attn_output)

            # Feed Forward
            residual = x
            x_norm = self.norm2(x)

            ff_output = self.mlp(x_norm)

            x = residual + self.dropout(ff_output)

        else:  # Post-Norm
            # Attention
            attn_output, attn_weights = self.attn(x, x, x)

            x = x + self.dropout(attn_output)
            x = self.norm1(x)

            # Feed Forward
            ff_output = self.mlp(x)

            x = x + self.dropout(ff_output)
            x = self.norm2(x)

        return x, attn_weights


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, depth, act, mlp_ratio, dropout, pre_norm):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    act=act,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    pre_norm=pre_norm,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x):
        attention_weights = []
        for layer in self.layers:
            x, attn = layer(x)
            attention_weights.append(attn)

        return x, attention_weights
