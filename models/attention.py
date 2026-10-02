import torch
import torch.nn as nn
from typing import Tuple
import math
import numpy as np
from math import sqrt


class SlidingWindowAttention(nn.Module):
    """
    Self-attention cho local/short-term context.
    Input: x: [B, L, D]
    Output: y: [B, L, D]
    """

    def __init__(self, d_model, num_heads, window_size=32, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.window_size = window_size

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, x):
        B, L, D = x.shape

        # --------------------------------------------------
        # Local attention mask
        # --------------------------------------------------
        device = x.device

        idx = torch.arange(L, device=device)

        distance = idx[:, None] - idx[None, :]

        # chỉ cho phép attention trong window
        mask = distance.abs() > self.window_size

        # MultiheadAttention:
        # True = không được attend
        y, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)

        return y


class WinAttention(nn.Module):
    def __init__(self, d_model, nhead, window_size=12, dropout=0.1):
        super().__init__()
        self.window_size = window_size
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

    def forward(self, x, y=None, z=None):
        # x: [B, L, D]
        B, L, D = x.shape
        w = self.window_size

        x = x.view(B, -1, w, D)  # [B, num_win, w, D]
        x = x.reshape(-1, w, D)  # [B*num_win, w, D]

        out, attn_weights = self.attn(x, x, x)

        out = out.reshape(B, -1, w, D).reshape(B, -1, D)
        return out, attn_weights


class CrossAttention(nn.Module):
    def __init__(self, x_dim, y_dim, num_heads, dropout, **kwargs):
        super().__init__()

        d_model = (x_dim + y_dim) // 2

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})."
            )

        self.x_proj = nn.Identity() if x_dim == d_model else nn.Linear(x_dim, d_model)
        self.y_proj = nn.Identity() if y_dim == d_model else nn.Linear(y_dim, d_model)

        self.x_to_y_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.y_to_x_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        self.x_proj_out = (
            nn.Identity() if d_model == x_dim else nn.Linear(d_model, x_dim)
        )
        self.y_proj_out = (
            nn.Identity() if d_model == y_dim else nn.Linear(d_model, y_dim)
        )

    def forward(self, x, y) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, Tx, x_dim)
        # y: (B, Ty, y_dim)

        x_proj = self.x_proj(x)
        y_proj = self.y_proj(y)

        x_cross, _ = self.x_to_y_cross_attn(x_proj, y_proj, y_proj)
        y_cross, _ = self.y_to_x_cross_attn(y_proj, x_proj, x_proj)

        x_out = self.x_proj_out(x_cross)
        y_out = self.y_proj_out(y_cross)

        return x_out, y_out


class ProbMask:
    """
    Source from: https://github.com/zhouhaoyi/Informer2020/blob/main/utils/masking.py#L13
    """

    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[
            torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :
        ].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask


class ProbAttention(nn.Module):
    """
    Source from: https://github.com/zhouhaoyi/Informer2020/blob/main/models/attn.py#L38
    """

    def __init__(
        self, mask_flag=True, factor=5, scale=None, attn_dropout=0.1, output_attn=False
    ):
        super().__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attn
        self.dropout = nn.Dropout(attn_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(
            -2
        )

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[
            torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], M_top, :
        ]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert L_Q == L_V  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[
            torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :
        ] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[
                torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :
            ] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype("int").item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype("int").item()  # c*ln(L_q)

        U_part = min(L_K, U_part)
        u = min(L_Q, u)

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1.0 / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask
        )

        return context.transpose(2, 1).contiguous(), attn
