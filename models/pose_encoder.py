import torch
import torch.nn as nn

from models.mlp import MLP
from models.transformer import TransformerEncoder


class PoseEncoder(nn.Module):
    """
    Transformer-based encoder for pose sequences.

    Input:
        x: (B, T, input_dim)

    Output:
        x: (B, T, d_model)
    """

    def __init__(
        self,
        in_dim,
        embed_dim,
        num_heads,
        depth,
        mlp_ratio,
        dropout,
        act,
        max_frames,
        pre_norm,
    ):
        super().__init__()

        # Project pose features to Transformer dimension
        self.input_proj = nn.Linear(in_dim, embed_dim)

        # Learnable positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, max_frames, embed_dim) * 0.02)

        # Transformer Encoder
        self.transformer = TransformerEncoder(
            d_model=embed_dim,
            num_heads=num_heads,
            depth=depth,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            act=act,
            pre_norm=pre_norm,
        )

        self.norm = nn.LayerNorm(embed_dim) if pre_norm else None

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Pose sequence.
            Shape: (B, T, input_dim)

        mask : torch.Tensor, optional
            Padding mask.
            Shape: (B, T)
            True = ignored position.

        Returns
        -------
        torch.Tensor
            Shape: (B, T, d_model)
        """

        B, T, _ = x.shape

        if T > self.pos_embedding.size(1):
            raise ValueError(
                f"Sequence length {T} exceeds max_seq_len {self.pos_embedding.size(1)}"
            )

        # (B, T, input_dim) -> (B, T, d_model)
        x = self.input_proj(x)

        # Add positional information
        x = x + self.pos_embedding[:, :T, :]

        # Transformer
        x, _ = self.transformer(x)

        # Final normalization
        if self.norm is not None:
            x = self.norm(x)

        return x
