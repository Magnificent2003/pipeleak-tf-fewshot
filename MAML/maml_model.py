from collections import OrderedDict
from typing import Dict

import torch
import torch.autograd as autograd
import torch.nn.functional as F

from .classifiers import LogisticClassifier
from .encoders import DarkNet19Encoder
from .modules import Module, get_child_dict


class MAMLClassifier(Module):
    """
    Minimal MAML model for 4-way classification:
    - encoder: DarkNet-19
    - classifier: linear logistic head
    """

    def __init__(self, n_way: int = 4):
        super().__init__()
        self.encoder = DarkNet19Encoder()
        self.classifier = LogisticClassifier(in_dim=self.encoder.get_out_dim(), n_way=n_way)

    def reset_classifier(self):
        self.classifier.reset_parameters()

    def _inner_forward(self, x: torch.Tensor, params: OrderedDict) -> torch.Tensor:
        feat = self.encoder(x, get_child_dict(params, "encoder"))
        logits = self.classifier(feat, get_child_dict(params, "classifier"))
        return logits

    def _adapt(
        self,
        x_shot: torch.Tensor,
        y_shot: torch.Tensor,
        params: OrderedDict,
        inner_args: Dict,
        meta_train: bool,
    ) -> OrderedDict:
        mom_buffer = OrderedDict()
        use_mom = float(inner_args["momentum"]) > 0.0
        if use_mom:
            for name, param in params.items():
                mom_buffer[name] = torch.zeros_like(param)

        for _ in range(int(inner_args["n_step"])):
            logits = self._inner_forward(x_shot, params)
            loss = F.cross_entropy(logits, y_shot)

            grads = autograd.grad(
                loss,
                params.values(),
                create_graph=(meta_train and not bool(inner_args["first_order"])),
                allow_unused=True,
            )

            updated = OrderedDict()
            for (name, param), grad in zip(params.items(), grads):
                if grad is None:
                    new_param = param
                else:
                    if float(inner_args["weight_decay"]) > 0.0:
                        grad = grad + float(inner_args["weight_decay"]) * param
                    if use_mom:
                        grad = grad + float(inner_args["momentum"]) * mom_buffer[name]
                        mom_buffer[name] = grad

                    if "encoder" in name:
                        lr = float(inner_args["encoder_lr"])
                    elif "classifier" in name:
                        lr = float(inner_args["classifier_lr"])
                    else:
                        raise ValueError(f"Unexpected parameter name: {name}")

                    new_param = param - lr * grad

                if not meta_train:
                    new_param = new_param.detach().requires_grad_(True)
                updated[name] = new_param
            params = updated
        return params

    def forward(
        self,
        x_shot: torch.Tensor,
        x_query: torch.Tensor,
        y_shot: torch.Tensor,
        inner_args: Dict,
        meta_train: bool,
    ) -> torch.Tensor:
        """
        Args:
          x_shot:  [B, n_way*n_support, C, H, W]
          x_query: [B, n_way*n_query, C, H, W]
          y_shot:  [B, n_way*n_support]
        Returns:
          logits:  [B, n_way*n_query, n_way]
        """
        if x_shot.dim() != 5 or x_query.dim() != 5:
            raise ValueError("x_shot/x_query must be 5D.")
        if x_shot.size(0) != x_query.size(0):
            raise ValueError("meta-batch mismatch between support/query.")

        params = OrderedDict(self.named_parameters())
        frozen_keys = list(inner_args.get("frozen", []))
        for name in list(params.keys()):
            if (not params[name].requires_grad) or any(fk in name for fk in frozen_keys + ["temp"]):
                params.pop(name)

        logits_all = []
        for ep in range(x_shot.size(0)):
            updated = self._adapt(
                x_shot=x_shot[ep],
                y_shot=y_shot[ep],
                params=params,
                inner_args=inner_args,
                meta_train=meta_train,
            )
            logits_ep = self._inner_forward(x_query[ep], updated)
            logits_all.append(logits_ep)
        logits = torch.stack(logits_all, dim=0)
        return logits
