import re
from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F


def get_child_dict(params, key=None):
    if params is None:
        return None
    if key is None or key == "":
        return params

    key_re = re.compile(r"^{0}\.(.+)".format(re.escape(key)))
    if not any(filter(key_re.match, params.keys())):
        key_re = re.compile(r"^module\.{0}\.(.+)".format(re.escape(key)))
    child_dict = OrderedDict(
        (key_re.sub(r"\1", k), value) for (k, value) in params.items() if key_re.match(k) is not None
    )
    return child_dict


class Module(nn.Module):
    def __init__(self):
        super().__init__()


class Conv2d(nn.Conv2d, Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x, params=None):
        if params is None:
            return super().forward(x)
        weight = params.get("weight", self.weight)
        bias = params.get("bias", self.bias)
        return F.conv2d(x, weight, bias, self.stride, self.padding)


class Linear(nn.Linear, Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)

    def forward(self, x, params=None):
        if params is None:
            return super().forward(x)
        weight = params.get("weight", self.weight)
        bias = params.get("bias", self.bias)
        return F.linear(x, weight, bias)


class BatchNorm2d(nn.BatchNorm2d, Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def forward(self, x, params=None):
        if params is None:
            return super().forward(x)
        weight = params.get("weight", self.weight)
        bias = params.get("bias", self.bias)
        return F.batch_norm(
            x,
            self.running_mean,
            self.running_var,
            weight,
            bias,
            self.training,
            self.momentum,
            self.eps,
        )
