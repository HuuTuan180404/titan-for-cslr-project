import torch.nn as nn


def get_activation(act: str) -> nn.Module:
    if act == "relu":
        return nn.ReLU()
    elif act == "gelu":
        return nn.GELU()
    elif act == "silu":
        return nn.SiLU()
    elif act == "swish":
        return nn.SiLU()
    else:
        raise ValueError(f"Unsupported activation: {act}")
