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


class ResBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + residual)


class ResNet1DClassifier(nn.Module):
    """Residual 1D network used for the classifier-generality experiment."""

    def __init__(self, input_dim: int, n_classes: int, dropout: float = 0.5) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.blocks = nn.Sequential(ResBlock1D(64), ResBlock1D(64), ResBlock1D(64))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        return self.head(self.blocks(self.stem(x)))


class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM used for the classifier-generality experiment."""

    def __init__(self, input_dim: int, n_classes: int, dropout: float = 0.5, hidden: int = 64) -> None:
        super().__init__()
        self.input_proj = nn.Linear(1, 32)
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        features = torch.relu(self.input_proj(x))
        outputs, _ = self.lstm(features)
        pooled = outputs.mean(dim=1)
        return self.classifier(pooled)


SUPPORTED_CLASSIFIERS = ("cnn1d", "resnet1d", "bilstm")


def build_classifier(
    architecture: str,
    input_dim: int,
    n_classes: int,
    dropout: float = 0.5,
) -> nn.Module:
    architecture = architecture.lower()
    if architecture == "cnn1d":
        return CNN1DClassifier(input_dim, n_classes, dropout=dropout)
    if architecture == "resnet1d":
        return ResNet1DClassifier(input_dim, n_classes, dropout=dropout)
    if architecture == "bilstm":
        return BiLSTMClassifier(input_dim, n_classes, dropout=dropout)
    raise ValueError(
        f"Unknown classifier architecture: {architecture}. "
        f"Choose one of: {', '.join(SUPPORTED_CLASSIFIERS)}"
    )


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
    architecture: str = "cnn1d",
) -> nn.Module:
    model = build_classifier(architecture, input_dim, n_classes, dropout=config.dropout).to(device)
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
            # BatchNorm1d cannot be trained on a singleton batch.
            if xb.size(0) < 2:
                continue
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
