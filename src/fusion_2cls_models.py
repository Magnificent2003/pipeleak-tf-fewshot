import torch
import torch.nn as nn

from Gatenet import Gatenet


def _safe_logit(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = torch.clamp(p, eps, 1.0 - eps)
    return torch.log(p) - torch.log(1.0 - p)


def _binary_entropy_from_prob(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = torch.clamp(p, eps, 1.0 - eps)
    return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))


def _positive_prob_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.clamp(torch.softmax(logits, dim=1)[:, 1], 1e-6, 1.0 - 1e-6)


class CWGFFusion2Cls(nn.Module):
    """
    CWGF-like 2-class fusion:
    - Build 5-dim hand-crafted features from expert positive probabilities.
    - GateNet predicts sample-wise expert weight g in (0,1).
    - Fuse in logit space on positive class probability.
    """

    def __init__(self, hid: int = 16, drop: float = 0.1):
        super().__init__()
        self.gate = Gatenet(in_dim=5, hid=hid, drop=drop)
        self.register_buffer("phi_mu", torch.zeros(1, 5))
        self.register_buffer("phi_std", torch.ones(1, 5))
        self._init_equal_weight()

    def _init_equal_weight(self, small: float = 1e-3) -> None:
        last_lin = None
        for m in self.gate.modules():
            if isinstance(m, nn.Linear):
                last_lin = m
        if last_lin is None:
            return
        with torch.no_grad():
            nn.init.normal_(last_lin.weight, mean=0.0, std=small)
            if last_lin.bias is not None:
                last_lin.bias.zero_()

    def set_feature_norm(self, mu: torch.Tensor, std: torch.Tensor) -> None:
        if mu.ndim == 1:
            mu = mu.unsqueeze(0)
        if std.ndim == 1:
            std = std.unsqueeze(0)
        if mu.shape != (1, 5) or std.shape != (1, 5):
            raise ValueError(f"mu/std shape must be [1,5], got {mu.shape}/{std.shape}")
        self.phi_mu.copy_(mu)
        self.phi_std.copy_(std)

    def fit_feature_norm_from_logits(self, logits_res: torch.Tensor, logits_dark: torch.Tensor) -> None:
        with torch.no_grad():
            phi = self._build_phi(logits_res, logits_dark)
            mu = phi.mean(dim=0, keepdim=True)
            std = phi.std(dim=0, keepdim=True) + 1e-6
            self.set_feature_norm(mu, std)

    def _build_phi(self, logits_res: torch.Tensor, logits_dark: torch.Tensor) -> torch.Tensor:
        p_res = _positive_prob_from_logits(logits_res)
        p_dark = _positive_prob_from_logits(logits_dark)
        return torch.stack(
            [
                p_res,
                p_dark,
                torch.abs(p_res - p_dark),
                _binary_entropy_from_prob(p_res),
                _binary_entropy_from_prob(p_dark),
            ],
            dim=1,
        )

    def forward(self, logits_res: torch.Tensor, logits_dark: torch.Tensor):
        p_res = _positive_prob_from_logits(logits_res)
        p_dark = _positive_prob_from_logits(logits_dark)

        phi = self._build_phi(logits_res, logits_dark)
        phi_n = (phi - self.phi_mu) / (self.phi_std + 1e-6)
        g_res = self.gate(phi_n)

        p_fuse = torch.sigmoid(g_res * _safe_logit(p_res) + (1.0 - g_res) * _safe_logit(p_dark))
        weights = torch.stack([g_res, 1.0 - g_res], dim=1)
        return p_fuse, {"expert_weights": weights, "phi": phi_n}


class AttentionFusion2Cls(nn.Module):
    """
    Sample-wise, class-agnostic attention fusion.
    """

    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.proj_res = nn.Linear(2, hidden_dim)
        self.proj_dark = nn.Linear(2, hidden_dim)
        self.score_res = nn.Linear(hidden_dim, 1)
        self.score_dark = nn.Linear(hidden_dim, 1)
        self._init_equal_weight()

    def _init_equal_weight(self) -> None:
        nn.init.xavier_uniform_(self.proj_res.weight)
        nn.init.xavier_uniform_(self.proj_dark.weight)
        nn.init.zeros_(self.proj_res.bias)
        nn.init.zeros_(self.proj_dark.bias)

        nn.init.zeros_(self.score_res.weight)
        nn.init.zeros_(self.score_dark.weight)
        nn.init.zeros_(self.score_res.bias)
        nn.init.zeros_(self.score_dark.bias)

    def forward(self, logits_res: torch.Tensor, logits_dark: torch.Tensor):
        h_res = torch.relu(self.proj_res(logits_res))
        h_dark = torch.relu(self.proj_dark(logits_dark))

        s_res = self.score_res(torch.tanh(h_res))
        s_dark = self.score_dark(torch.tanh(h_dark))

        alpha = torch.softmax(torch.cat([s_res, s_dark], dim=1), dim=1)
        fused_logits = alpha[:, 0:1] * logits_res + alpha[:, 1:2] * logits_dark
        p_fuse = _positive_prob_from_logits(fused_logits)
        return p_fuse, {"expert_weights": alpha, "fused_logits": fused_logits}


class MoEStyleFusion2Cls(nn.Module):
    """
    Lightweight soft MoE baseline:
    gate input u = [z_res; z_dark; maxprob_res; maxprob_dark; entropy_res; entropy_dark]
    dims: 8 -> 16 -> 2.
    """

    def __init__(self, hidden_dim: int = 16, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(8, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 2)
        self._init_equal_weight()

    def _init_equal_weight(self) -> None:
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def _stats_from_logits(self, logits: torch.Tensor):
        probs = torch.softmax(logits, dim=1)
        maxprob = probs.max(dim=1).values
        entropy = -(probs * torch.log(torch.clamp(probs, 1e-9, 1.0))).sum(dim=1)
        return maxprob, entropy

    def forward(self, logits_res: torch.Tensor, logits_dark: torch.Tensor):
        max_res, ent_res = self._stats_from_logits(logits_res)
        max_dark, ent_dark = self._stats_from_logits(logits_dark)

        u = torch.cat(
            [
                logits_res,
                logits_dark,
                max_res.unsqueeze(1),
                max_dark.unsqueeze(1),
                ent_res.unsqueeze(1),
                ent_dark.unsqueeze(1),
            ],
            dim=1,
        )

        g_logits = self.fc2(self.drop(torch.relu(self.fc1(u))))
        g = torch.softmax(g_logits, dim=1)
        fused_logits = g[:, 0:1] * logits_res + g[:, 1:2] * logits_dark
        p_fuse = _positive_prob_from_logits(fused_logits)
        return p_fuse, {"expert_weights": g, "fused_logits": fused_logits, "gate_logits": g_logits}


def build_fusion_2cls(method: str, hidden_dim: int = 16, dropout: float = 0.1) -> nn.Module:
    m = method.lower().strip()
    if m == "cwgf":
        return CWGFFusion2Cls(hid=hidden_dim, drop=dropout)
    if m == "attention":
        return AttentionFusion2Cls(hidden_dim=hidden_dim)
    if m == "moe":
        return MoEStyleFusion2Cls(hidden_dim=hidden_dim, dropout=dropout)
    raise ValueError(f"Unsupported fusion method: {method}")
