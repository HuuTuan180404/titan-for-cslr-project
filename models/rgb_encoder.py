import torch
import torch.nn as nn
from .transformer import TransformerEncoderLayer


class PatchEmbedding(nn.Module):
    """
    Image -> Patch tokens

    Input:
        x: (B, C, H, W)

    Output:
        x: (B, N, D)

    B: Batch size
    C: Number of channels
    H: Image height
    W: Image width
    N: Number of patches
    D: Embedding dimension
    """

    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super().__init__()

        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.image_size = image_size
        self.patch_size = patch_size

        self.num_patches = (image_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x) -> torch.Tensor:
        # (B, C, H, W)
        x = self.proj(x)

        # (B, D, H/P, W/P)
        x = x.flatten(2)

        # (B, N, D)
        x = x.transpose(1, 2)

        return x


class VisualEncoder(nn.Module):
    """
    Vision Transformer.

    Each frame is processed independently by ViT.

    Input:
        x: (B, L, C, H, W)

    Output:
        x: (B, L, N+1, D)

    B: Batch size
    L: Temporal length / number of frames
    C: RGB channels
    H: Image height
    W: Image width
    N: Number of patches
    D: Embedding dimension
    """

    def __init__(
        self,
        image_size,
        patch_size,
        in_channels,
        act,
        embed_dim,
        depth,
        num_heads,
        mlp_ratio,
        dropout,
        pre_norm,
    ):
        super().__init__()
        self.count_layer = 0
        self.B = 0
        self.L = 0

        # ==========================================
        # 1. Patch Embedding
        # ==========================================
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # ==========================================
        # 2. CLS Token
        # ==========================================
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # ==========================================
        # 3. Positional Embedding
        # ==========================================
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)

        # ==========================================
        # 4. Transformer Encoder
        # ==========================================
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=embed_dim,
                    num_heads=num_heads,
                    act=act,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    pre_norm=pre_norm,
                )
                for _ in range(depth)
            ]
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x) -> torch.Tensor:
        if self.count_layer == 0:
            # (B, L, C, H, W)
            B, L, C, H, W = x.shape
            self.B, self.L = B, L

            # (B, L, C, H, W) -> (B*L, C, H, W)
            x = x.reshape(B * L, C, H, W)

            # (B*L, C, H, W) -> (B*L, N, D)
            x = self.patch_embed(x)

            # (1, 1, D) -> (B*L, 1, D)
            cls_token = self.cls_token.expand(B * L, -1, -1)

            # (B*L, N, D) + (B*L, 1, D) -> (B*L, N+1, D)
            x = torch.cat((cls_token, x), dim=1)

            # (B*L, N+1, D)
            x = x + self.pos_embed

            x = self.pos_dropout(x)

        x, _ = self.blocks[self.count_layer](x)  # (B*L, N+1, D)
        self.count_layer += 1

        if self.count_layer == len(self.blocks):
            # (B*L, N+1, D) -> (B, L, N+1, D)
            x = x.reshape(self.B, self.L, x.shape[1], x.shape[2])

        return x
