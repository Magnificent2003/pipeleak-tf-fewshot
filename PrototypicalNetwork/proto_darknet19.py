from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_c),
        nn.LeakyReLU(0.1, inplace=True),
    )


def _euclidean_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # x: [N, D], y: [M, D] -> [N, M]
    n, m = x.size(0), y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise ValueError(f"Embedding dims mismatch: {d} vs {y.size(1)}")
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return (x - y).pow(2).sum(dim=2)


class DarkNet19Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(3, 32),
            nn.MaxPool2d(2),
            _conv_block(32, 64),
            nn.MaxPool2d(2),
            _conv_block(64, 128),
            _conv_block(128, 64, k=1, s=1, p=0),
            _conv_block(64, 128),
            nn.MaxPool2d(2),
            _conv_block(128, 256),
            _conv_block(256, 128, k=1, s=1, p=0),
            _conv_block(128, 256),
            nn.MaxPool2d(2),
            _conv_block(256, 512),
            _conv_block(512, 256, k=1, s=1, p=0),
            _conv_block(256, 512),
            _conv_block(512, 256, k=1, s=1, p=0),
            _conv_block(256, 512),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return x.view(x.size(0), -1)


class ProtoDarkNet19(nn.Module):
    """
    4-class prototypical network using DarkNet-19 encoder.
    Forward path is episode-based (support/query).
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder = DarkNet19Encoder()

    def forward_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def episode_logits(self, xs: torch.Tensor, xq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        xs: [n_way, n_support, C, H, W]
        xq: [n_way, n_query, C, H, W]
        returns:
          log_p_y: [n_way, n_query, n_way]
          target: [n_way, n_query]
        """
        n_way = xs.size(0)
        if xq.size(0) != n_way:
            raise ValueError("Support/query n_way mismatch.")
        n_support = xs.size(1)
        n_query = xq.size(1)

        x = torch.cat(
            [
                xs.view(n_way * n_support, *xs.size()[2:]),
                xq.view(n_way * n_query, *xq.size()[2:]),
            ],
            dim=0,
        )
        z = self.forward_embedding(x)
        z_dim = z.size(-1)

        z_proto = z[: n_way * n_support].view(n_way, n_support, z_dim).mean(dim=1)
        z_query = z[n_way * n_support :]

        dists = _euclidean_dist(z_query, z_proto)
        log_p_y = F.log_softmax(-dists, dim=1).view(n_way, n_query, n_way)

        target = (
            torch.arange(0, n_way, device=xs.device)
            .view(n_way, 1)
            .expand(n_way, n_query)
            .long()
        )
        return log_p_y, target

    def episode_loss(self, xs: torch.Tensor, xq: torch.Tensor):
        log_p_y, target = self.episode_logits(xs, xq)
        loss = -log_p_y.gather(2, target.unsqueeze(-1)).squeeze(-1).contiguous().view(-1).mean()
        pred = log_p_y.argmax(dim=2)
        acc = (pred == target).float().mean()
        return loss, float(acc.item()), target.reshape(-1), pred.reshape(-1)
