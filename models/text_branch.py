import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.modules.mamba_simple import Mamba

from models.utils import get_activation
from models.mlp import MLP

"""
gloss_ids [B, L]
      │
      ▼
   Embedding
      │
      ▼
   Mamba × N
      │
      ▼
token_features [B, L, D]
      │
      │ masked mean pooling
      ▼
sentence_features [B, D]
      │
      │ projection head
      ▼
z_text [B, E]
"""


class MambaTextEncoder(nn.Module):
    """
    Mamba-based text encoder for visual-textual contrastive learning.

    Input:
        input_ids: [B, L]
        attention_mask: [B, L]

    Output:
        token_features: [B, L, D]
        sentence_features: [B, D]
    """

    def __init__(
        self,
        vocab_size,
        embed_dim,
        depth,
        state_dim,
        conv_kernel,
        expand,
        dropout,
        padding_idx,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx

        # --------------------------------------------------
        # 1. Token Embedding
        # --------------------------------------------------
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size + 2,
            embedding_dim=embed_dim,
            padding_idx=padding_idx,
        )

        self.embedding_dropout = nn.Dropout(dropout)

        # --------------------------------------------------
        # 2. Mamba blocks
        # --------------------------------------------------
        self.mamba_layers = nn.ModuleList(
            [
                Mamba(
                    d_model=embed_dim,
                    d_state=state_dim,
                    d_conv=conv_kernel,
                    expand=expand,
                )
                for _ in range(depth)
            ]
        )

        # LayerNorm after each Mamba block
        self.norm_layers = nn.ModuleList(
            [nn.LayerNorm(embed_dim) for _ in range(depth)]
        )

        self.dropout = nn.Dropout(dropout)

        # --------------------------------------------------
        # 3. Final normalization
        # --------------------------------------------------
        self.final_norm = nn.LayerNorm(embed_dim)

    def masked_mean_pooling(self, x, attention_mask) -> torch.Tensor:
        """
        x: [B, L, D]
        attention_mask: [B, L]
            1 = valid token
            0 = padding
        """

        mask = attention_mask.unsqueeze(-1).float()
        # [B, L, 1]

        x = x * mask

        summed = x.sum(dim=1)
        # [B, D]

        count = mask.sum(dim=1).clamp(min=1e-6)
        # [B, 1]

        return summed / count

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: [B, L]
            attention_mask: [B, L]
        Returns:
            token_features: [B, L, D]
            sentence_features: [B, D]
        """

        # --------------------------------------------------
        # Token embedding
        # --------------------------------------------------
        x = self.embedding(input_ids)
        # [B, L, D]

        x = self.embedding_dropout(x)

        # --------------------------------------------------
        # Zero-out padding tokens
        # --------------------------------------------------
        mask = attention_mask.unsqueeze(-1).float()
        x = x * mask  # [B, L, D]

        # --------------------------------------------------
        # Mamba blocks
        # --------------------------------------------------
        for mamba, norm in zip(self.mamba_layers, self.norm_layers):
            residual = x

            x = mamba(x)
            # [B, L, D]

            x = self.dropout(x)

            x = residual + x

            x = norm(x)

            # Keep padding positions zero
            x = x * mask

        # --------------------------------------------------
        # Final normalization
        # --------------------------------------------------
        x = self.final_norm(x)

        # --------------------------------------------------
        # Sentence-level representation
        # --------------------------------------------------
        sentence_features = self.masked_mean_pooling(x, attention_mask)
        # [B, D]

        return x, sentence_features


class TextBranch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = MambaTextEncoder(
            vocab_size=config.model.t_branch.vocab_size,
            embed_dim=config.model.t_branch.embed_dim,
            depth=config.model.t_branch.depth,
            padding_idx=config.model.t_branch.padding_idx,
            state_dim=config.model.t_branch.state_dim,
            conv_kernel=config.model.t_branch.conv_kernel,
            expand=config.model.t_branch.expand,
            dropout=config.model.dropout,
        )

        self.projection = MLP(
            in_dim=config.model.t_branch.embed_dim,
            mlp_ratio=2,
            act=config.model.act,
            dropout=config.model.dropout,
            out_dim=config.model.t_branch.proj_dim,
        )

    def forward(self, input_ids, attention_mask):

        token_features, sentence_features = self.encoder.forward(
            input_ids=input_ids, attention_mask=attention_mask
        )

        z_text = self.projection(sentence_features)  # [B, D]
        z_text = F.normalize(z_text, dim=-1)

        return {
            "token_features": token_features,
            "sentence_features": sentence_features,
            "z_text": z_text,
        }
