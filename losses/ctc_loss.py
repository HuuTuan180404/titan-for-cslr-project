import torch.nn as nn


class CTCLoss(nn.Module):
    def __init__(self, blank, zero_infinity=True):
        super().__init__()
        self.loss = nn.CTCLoss(blank=blank, zero_infinity=zero_infinity)

    def forward(self, logits, targets, input_lengths, target_lengths):
        # logits: [B, T, C]

        log_probs = logits.log_softmax(dim=-1)

        # [B, T, C] -> [T, B, C]
        log_probs = log_probs.transpose(0, 1)

        return self.loss(log_probs, targets, input_lengths, target_lengths)
