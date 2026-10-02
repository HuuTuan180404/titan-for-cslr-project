import torch
import torch.nn as nn
import torch.nn.functional as F

from .pose_branch import PoseBranch
from .text_branch import TextBranch

from utils.utils import get_config
from models.utils import get_activation
from models.mlp import MLP


class DualBranchCSLRModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.config = cfg
        dropout = cfg.model.dropout
        act = cfg.model.act

        # ------------------------------------------------------------
        # 1. Image branch
        # ------------------------------------------------------------
        self.image_encoder = PoseBranch(cfg)
        self.full_pose_dim = self.image_encoder.full_pose_dim

        # ------------------------------------------------------------
        # 2. Text branch
        # ------------------------------------------------------------
        self.vocab_size = cfg.model.t_branch.vocab_size
        # CTC classes: PAD (0), glosses (1..vocab_size), UNK, then BLANK.
        self.ctc_num_classes = self.vocab_size + 3
        self.text_projection_dim = cfg.model.t_branch.proj_dim

        self.text_encoder = TextBranch(cfg)

        # ------------------------------------------------------------
        # 3. Projection head cho visual branch
        # ------------------------------------------------------------
        self.image_proj = MLP(
            in_dim=cfg.model.p_branch.embed_dim,
            mlp_ratio=2,
            act=act,
            dropout=dropout,
            out_dim=cfg.model.p_branch.proj_dim,
        )

        # ------------------------------------------------------------
        # 4. Classifier / fusion head (phác thảo)
        # ------------------------------------------------------------
        self.ctc_logits = MLP(
            in_dim=cfg.model.p_branch.embed_dim,
            mlp_ratio=2,
            act=act,
            dropout=dropout,
            out_dim=self.ctc_num_classes,
        )

        self.contrastive_logit_scale = nn.Parameter(
            torch.log(torch.tensor(1 / cfg.loss.contrastive.temperature))
        )

    def forward(
        self,
        lh_pose,
        rh_pose,
        lh_rgb=None,
        rh_rgb=None,
        face=None,
        body=None,
        gloss_ids=None,
        text_attention_mask=None,
    ):
        # ------------------------------------------------------------
        # 1. Visual features
        # ------------------------------------------------------------
        image_features = self.image_encoder.forward(
            lh_pose, rh_pose, lh_rgb, rh_rgb, face, body
        )  # (B, T, embed_dim)
        ctc_logits = self.ctc_logits(image_features)

        # Pool temporal dimension để lấy biểu diễn toàn video
        image_context = image_features.mean(dim=1)  # [B, D_img]
        z_img = self.image_proj(image_context)  # [B, proj_dim]
        z_img = F.normalize(z_img, dim=-1)

        # ------------------------------------------------------------
        # 2. Text features
        # ------------------------------------------------------------
        z_text = None
        if gloss_ids is not None and text_attention_mask is not None:
            text_dict = self.text_encoder.forward(
                input_ids=gloss_ids, attention_mask=text_attention_mask
            )
            z_text = text_dict["z_text"]  # [B, proj_text]

        contrastive_logits = None
        if z_text is not None:
            similarity = z_img @ z_text.T
            contrastive_logits = similarity * self.contrastive_logit_scale.exp()

        return {
            "contrastive_logits": contrastive_logits,
            "ctc_logits": ctc_logits,
        }


# Alias ngắn gọn để tiện import
CSLRModel = DualBranchCSLRModel
