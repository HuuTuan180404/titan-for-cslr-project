import torch.nn as nn
from models.utils import get_activation


class MLP(nn.Module):
    def __init__(self, in_dim, mlp_ratio, act, dropout, hidden_dim=None, out_dim=None):
        super().__init__()

        hidden_dim = mlp_ratio * in_dim if hidden_dim is None else hidden_dim
        out_dim = out_dim if out_dim is not None else in_dim

        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.activation = get_activation(act)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
