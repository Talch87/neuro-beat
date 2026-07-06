import numpy as np
import torch

from neurocardio.eval.metrics import aami_metrics, confusion


def evaluate(model, loader, classes, device: str = "cpu") -> dict:
    model.to(device)
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            preds = model(x.to(device)).argmax(dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(np.asarray(y).tolist())
    cm = confusion(y_true, y_pred, n_classes=len(classes))
    return {"confusion": cm, "metrics": aami_metrics(cm, classes)}
