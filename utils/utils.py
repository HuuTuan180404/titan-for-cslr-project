import torch.nn as nn


def get_config(cfg, *keys):
    """Đọc giá trị từ OmegaConf / dict / object linh hoạt."""
    cur = cfg
    for key in keys:
        if isinstance(cur, dict):
            cur = cur.get(key)
        else:
            cur = getattr(cur, key)
        if cur is None:
            raise KeyError(f"Config key not found: {'.'.join(keys)}")
    return cur


# def get_activation(act: str) -> nn.Module:
#     if act == "relu":
#         return nn.ReLU()
#     elif act == "gelu":
#         return nn.GELU()
#     elif act == "silu":
#         return nn.SiLU()
#     elif act == "swish":
#         return nn.SiLU()
#     else:
#         raise ValueError(f"Unsupported activation: {act}")
