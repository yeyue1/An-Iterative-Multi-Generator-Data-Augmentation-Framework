from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..config import ClassifierConfig
from ..metrics import classification_metrics


class CNN1DClassifier(nn.Module):
    def __init__(self, input_dim: int, n_classes: int, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        return self.classifier(self.features(x))


def _make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def predict_proba(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    model.eval()
    probs: list[np.ndarray] = []
    loader = DataLoader(torch.as_tensor(x, dtype=torch.float32), batch_size=batch_size)
    with torch.no_grad():
        for xb in loader:
            logits = model(xb.to(device))
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probs, axis=0)


def train_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    n_classes: int,
    config: ClassifierConfig,
    device: torch.device,
    epochs: int | None = None,
) -> CNN1DClassifier:
    model = CNN1DClassifier(input_dim, n_classes, dropout=config.dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs or config.epochs)
    )
    criterion = nn.CrossEntropyLoss()
    train_loader = _make_loader(x_train, y_train, config.batch_size, shuffle=True)

    best_state = deepcopy(model.state_dict())
    best_f1 = -1.0
    no_improve = 0
    max_epochs = epochs or config.epochs

    for _ in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        val_prob = predict_proba(model, x_val, device)
        val_pred = val_prob.argmax(axis=1)
        val_f1 = classification_metrics(y_val, val_pred, val_prob, n_classes)["f1_macro"]
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= config.patience:
            break

    model.load_state_dict(best_state)
    return model
