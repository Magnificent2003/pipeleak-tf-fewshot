from typing import Tuple, Dict

import torch
from sklearn.metrics import f1_score

# --------------------- Metrics ---------------------
@torch.no_grad()
def metrics_from_logits(logits: torch.Tensor, y: torch.Tensor, num_classes: int) -> Dict[str, float]:
    pred = logits.argmax(dim=1)
    acc = float((pred == y).float().mean().item())

    y_np = y.cpu().numpy()
    p_np = pred.cpu().numpy()
    avg = "binary" if num_classes == 2 else "macro"
    f1 = float(f1_score(y_np, p_np, average=avg, zero_division=0))
    
    return {"acc": acc, "f1": f1}

def evaluate(model, loader, criterion, device, num_classes: int) -> Tuple[float, Dict[str, float]]:
    model.eval()
    loss_sum, n = 0.0, 0
    all_logits, all_y = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = torch.as_tensor(y, dtype=torch.long, device=device)
            out = model(x)
            loss = criterion(out, y)
            loss_sum += float(loss.item()) * x.size(0)
            n += x.size(0)
            all_logits.append(out)
            all_y.append(y)
    logits = torch.cat(all_logits, dim=0)
    ys = torch.cat(all_y, dim=0)
    mets = metrics_from_logits(logits, ys, num_classes)
    return loss_sum / max(n,1), mets

# --------------------- Train One Epoch ---------------------
def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    loss_sum, n = 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = torch.as_tensor(y, dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.item()) * x.size(0)
        n += x.size(0)
    return loss_sum / max(n,1)
