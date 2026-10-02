import torch
import torch.nn as nn
from typing import Tuple

from .rgb_encoder import VisualEncoder
from models.mlp import MLP
from .attention import CrossAttention


class HandEncoderLayer(nn.Module):
    def __init__(
        self,
        pose_embed_dim,
        pose_num_heads,
        rgb_embed_dim,
        act,
        mlp_ratio,
        dropout,
        pose_rgb_num_heads,
        pose_pose_num_heads,
        use_rgb,
        pre_norm,
    ):
        super().__init__()
        self.pre_norm = pre_norm

        # ============================================================
        # Pose <-> RGB Cross Attention
        # ============================================================
        self.lh_rgb_pose_cross = (
            CrossAttention(
                x_dim=pose_embed_dim,
                y_dim=rgb_embed_dim,
                num_heads=pose_rgb_num_heads,
                dropout=dropout,
            )
            if use_rgb
            else None
        )
        self.rh_rgb_pose_cross = (
            CrossAttention(
                x_dim=pose_embed_dim,
                y_dim=rgb_embed_dim,
                num_heads=pose_rgb_num_heads,
                dropout=dropout,
            )
            if use_rgb
            else None
        )

        # ============================================================
        # 1. Pose Self-Attention
        # ============================================================

        # Left hand
        self.lh_norm1 = nn.LayerNorm(pose_embed_dim)

        self.lh_attn = nn.MultiheadAttention(
            pose_embed_dim, pose_num_heads, dropout, batch_first=True
        )

        self.lh_attn_dropout = nn.Dropout(dropout)

        # Right hand
        self.rh_norm1 = nn.LayerNorm(pose_embed_dim)

        self.rh_attn = nn.MultiheadAttention(
            pose_embed_dim, pose_num_heads, dropout, batch_first=True
        )

        self.rh_attn_dropout = nn.Dropout(dropout)

        # ============================================================
        # 2. Left Pose <-> Right Pose Cross Attention
        # ============================================================

        self.lh_pose_rh_pose_attn = CrossAttention(
            x_dim=pose_embed_dim,
            y_dim=pose_embed_dim,
            num_heads=pose_pose_num_heads,
            dropout=dropout,
        )

        # Norm 2
        self.lh_norm2 = nn.LayerNorm(pose_embed_dim)
        self.rh_norm2 = nn.LayerNorm(pose_embed_dim)

        # Dropout for cross attention
        self.lh_cross_dropout = nn.Dropout(dropout)
        self.rh_cross_dropout = nn.Dropout(dropout)

        # ============================================================
        # 3. Feed Forward Network
        # ============================================================
        self.lh_mlp = MLP(
            in_dim=pose_embed_dim, mlp_ratio=mlp_ratio, act=act, dropout=dropout
        )
        self.rh_mlp = MLP(
            in_dim=pose_embed_dim, mlp_ratio=mlp_ratio, act=act, dropout=dropout
        )

        # Norm 3
        self.lh_norm3 = nn.LayerNorm(pose_embed_dim)
        self.rh_norm3 = nn.LayerNorm(pose_embed_dim)

        # Dropout
        self.lh_mlp_dropout = nn.Dropout(dropout)
        self.rh_mlp_dropout = nn.Dropout(dropout)

    def forward(self, lh_pose, rh_pose, lh_rgb, rh_rgb):
        B, L, D_p = lh_pose.shape
        # _, _, N, D_r = lh_rgb.shape if lh_rgb is not None else (0, 0, 0, 0)
        # ============================================================
        # Pose <-> RGB Cross Attention
        # ============================================================
        if self.lh_rgb_pose_cross is not None:
            # lh_rgb = lh_rgb.reshape(B * L, N, D_r)
            lh_pose = lh_pose.reshape(B * L, 1, D_p)

            lh_pose, lh_rgb = self.lh_rgb_pose_cross(lh_pose, lh_rgb)

            # lh_rgb = lh_rgb.reshape(B, L, N, D_r)
            lh_pose = lh_pose.reshape(B, L, D_p)

        if self.rh_rgb_pose_cross is not None:
            # rh_rgb = rh_rgb.reshape(B * L, N, D_r)
            rh_pose = rh_pose.reshape(B * L, 1, D_p)

            rh_pose, rh_rgb = self.rh_rgb_pose_cross(rh_pose, rh_rgb)

            # rh_rgb = rh_rgb.reshape(B, L, N, D_r)
            rh_pose = rh_pose.reshape(B, L, D_p)

        if self.pre_norm:  # PRE-NORM
            # ========================================================
            # 1. Pose Self-Attention
            # ========================================================
            # LH
            lh_residual = lh_pose

            lh_norm = self.lh_norm1(lh_pose)

            lh_attn_out, _ = self.lh_attn(lh_norm, lh_norm, lh_norm)

            lh_attn_out = self.lh_attn_dropout(lh_attn_out)

            lh_attn_out = lh_residual + lh_attn_out

            # RH
            rh_residual = rh_pose

            rh_norm = self.rh_norm1(rh_pose)

            rh_attn_out, _ = self.rh_attn(rh_norm, rh_norm, rh_norm)

            rh_attn_out = self.rh_attn_dropout(rh_attn_out)

            rh_attn_out = rh_residual + rh_attn_out

            # ========================================================
            # 2. LH <-> RH Cross Attention
            # ========================================================
            lh_residual = lh_attn_out
            rh_residual = rh_attn_out

            lh_cross_out, rh_cross_out = self.lh_pose_rh_pose_attn(
                lh_attn_out, rh_attn_out
            )

            lh_cross_out = self.lh_cross_dropout(lh_cross_out)
            rh_cross_out = self.rh_cross_dropout(rh_cross_out)

            lh_attn_out = lh_residual + lh_cross_out
            rh_attn_out = rh_residual + rh_cross_out

            # ========================================================
            # 3. MLP
            # ========================================================
            lh_residual = lh_attn_out
            rh_residual = rh_attn_out

            lh_mlp_out = self.lh_mlp(self.lh_norm3(lh_attn_out))

            rh_mlp_out = self.rh_mlp(self.rh_norm3(rh_attn_out))

            lh_mlp_out = self.lh_mlp_dropout(lh_mlp_out)
            rh_mlp_out = self.rh_mlp_dropout(rh_mlp_out)

            lh_mlp_out = lh_residual + lh_mlp_out
            rh_mlp_out = rh_residual + rh_mlp_out

        else:  # POST-NORM
            # ========================================================
            # 1. Pose Self-Attention
            # ========================================================

            # LH
            lh_attn_out, _ = self.lh_attn(lh_pose, lh_pose, lh_pose)
            lh_attn_out = self.lh_attn_dropout(lh_attn_out)
            lh_attn_out = self.lh_norm1(lh_pose + lh_attn_out)

            # RH
            rh_attn_out, _ = self.rh_attn(rh_pose, rh_pose, rh_pose)
            rh_attn_out = self.rh_attn_dropout(rh_attn_out)
            rh_attn_out = self.rh_norm1(rh_pose + rh_attn_out)

            # ========================================================
            # 2. LH <-> RH Cross Attention
            # ========================================================
            lh_cross_out, rh_cross_out = self.lh_pose_rh_pose_attn(
                lh_attn_out, rh_attn_out
            )

            lh_cross_out = self.lh_cross_dropout(lh_cross_out)
            rh_cross_out = self.rh_cross_dropout(rh_cross_out)

            lh_attn_out = self.lh_norm2(lh_attn_out + lh_cross_out)

            rh_attn_out = self.rh_norm2(rh_attn_out + rh_cross_out)

            # ========================================================
            # 3. MLP
            # ========================================================
            lh_mlp_out = self.lh_mlp(lh_attn_out)
            rh_mlp_out = self.rh_mlp(rh_attn_out)

            lh_mlp_out = self.lh_mlp_dropout(lh_mlp_out)
            rh_mlp_out = self.rh_mlp_dropout(rh_mlp_out)

            lh_mlp_out = self.lh_norm3(lh_attn_out + lh_mlp_out)

            rh_mlp_out = self.rh_norm3(rh_attn_out + rh_mlp_out)

        return lh_mlp_out, rh_mlp_out, lh_rgb, rh_rgb


class HandEncoder(nn.Module):
    """
    Transformer-based encoder for pose sequences.

    Input:
        x: (B, T, input_dim)

    Output:
        x: (B, T, d_model)
    """

    def __init__(
        self,
        hand_pose_input_dim,
        pose_embed_dim,
        pose_num_heads,
        rgb_image_size,
        rgb_patch_size,
        rgb_in_channels,
        rgb_embed_dim,
        act,
        rgb_num_heads,
        depth,
        dropout,
        pose_rgb_num_heads,
        pose_pose_num_heads,
        mlp_ratio,
        use_rgb,
        pre_norm,
    ):

        super().__init__()

        self.lh_pose_proj = nn.Linear(hand_pose_input_dim, pose_embed_dim)
        self.rh_pose_proj = nn.Linear(hand_pose_input_dim, pose_embed_dim)

        self.layers = nn.ModuleList(
            [
                HandEncoderLayer(
                    pose_embed_dim=pose_embed_dim,
                    pose_num_heads=pose_num_heads,
                    rgb_embed_dim=rgb_embed_dim,
                    act=act,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    pose_rgb_num_heads=pose_rgb_num_heads,
                    pose_pose_num_heads=pose_pose_num_heads,
                    use_rgb=use_rgb,
                    pre_norm=pre_norm,
                )
                for _ in range(depth)
            ]
        )

        # ============================================================
        # RGB: Visual Stream
        # ============================================================
        self.lh_rgb = (
            VisualEncoder(
                image_size=rgb_image_size,
                patch_size=rgb_patch_size,
                in_channels=rgb_in_channels,
                act=act,
                embed_dim=rgb_embed_dim,
                depth=depth,
                num_heads=rgb_num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                pre_norm=pre_norm,
            )
            if use_rgb
            else None
        )

        self.rh_rgb = (
            VisualEncoder(
                image_size=rgb_image_size,
                patch_size=rgb_patch_size,
                in_channels=rgb_in_channels,
                act=act,
                embed_dim=rgb_embed_dim,
                depth=depth,
                num_heads=rgb_num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                pre_norm=pre_norm,
            )
            if use_rgb
            else None
        )

    def forward(self, lh_pose, rh_pose, lh_rgb, rh_rgb):

        lh_pose = self.lh_pose_proj(lh_pose)
        rh_pose = self.rh_pose_proj(rh_pose)

        for layer in self.layers:
            if self.lh_rgb is not None:
                lh_rgb = self.lh_rgb(lh_rgb)
            if self.rh_rgb is not None:
                rh_rgb = self.rh_rgb(rh_rgb)

            lh_pose, rh_pose, lh_rgb, rh_rgb = layer(lh_pose, rh_pose, lh_rgb, rh_rgb)

        return lh_pose, rh_pose
