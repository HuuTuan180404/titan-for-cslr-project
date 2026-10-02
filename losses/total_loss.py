import torch.nn as nn

from .ctc_loss import CTCLoss
from .contrastive_loss import ContrastiveLoss


class TotalLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        vocab_size = cfg.model.t_branch.vocab_size
        lambda_ctc = cfg.loss.lambda_ctc
        lambda_contrastive = cfg.loss.lambda_contrastive

        self.ctc_weight = lambda_ctc
        self.contrastive_weight = lambda_contrastive

        self.ctc_loss = CTCLoss(
            blank=vocab_size + 2, zero_infinity=cfg.loss.ctc.zero_infinity
        )

        self.contrastive_loss = ContrastiveLoss()

    def forward(self, outputs, target_ids, input_lengths, target_lengths):
        # ------------------------------------------------------------
        # CTC loss
        # ------------------------------------------------------------
        ctc_loss = self.ctc_loss(
            outputs["ctc_logits"], target_ids, input_lengths, target_lengths
        )

        # ------------------------------------------------------------
        # Contrastive loss
        # ------------------------------------------------------------
        contrastive_loss = self.contrastive_loss(outputs["contrastive_logits"])

        # ------------------------------------------------------------
        # Total loss
        # ------------------------------------------------------------
        total_loss = (
            self.ctc_weight * ctc_loss + self.contrastive_weight * contrastive_loss
        )

        return {
            "loss": total_loss,
            "ctc_loss": ctc_loss,
            "contrastive_loss": contrastive_loss,
        }
