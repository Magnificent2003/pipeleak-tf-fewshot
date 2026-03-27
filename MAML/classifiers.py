import torch
import torch.nn as nn

from .modules import Linear, Module, get_child_dict


class LogisticClassifier(Module):
    def __init__(self, in_dim: int, n_way: int, temp: float = 1.0, learn_temp: bool = False):
        super().__init__()
        self.in_dim = int(in_dim)
        self.n_way = int(n_way)
        self.linear = Linear(self.in_dim, self.n_way)
        self.learn_temp = bool(learn_temp)
        if self.learn_temp:
            self.temp = nn.Parameter(torch.tensor(float(temp)))
        else:
            self.temp = float(temp)

    def reset_parameters(self):
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, params=None):
        logits = self.linear(x, get_child_dict(params, "linear"))
        logits = logits * self.temp
        return logits
