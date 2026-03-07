"""
Financial LSTM with multi-head attention for time series forecasting.

Supports CUDA, MPS (Apple Silicon), and CPU. Designed for price prediction
with technical indicators as features.
"""

from __future__ import annotations

import numpy as np
import structlog
import torch
import torch.nn as nn

from gpu.device import get_device

log = structlog.get_logger()


class FinancialLSTM(nn.Module):
    """
    Bidirectional LSTM with multi-head self-attention and residual connections.

    Architecture:
        Input → LSTM(3 layers, bidirectional) → MultiHeadAttention → FC → Output
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
        output_size: int = 1,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )

        self.attention = nn.MultiheadAttention(
            hidden_size * 2,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.layer_norm = nn.LayerNorm(hidden_size * 2)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        normed = self.layer_norm(attn_out + lstm_out)
        return self.fc(normed[:, -1, :])


class LSTMForecaster:
    """High-level forecaster using FinancialLSTM."""

    def __init__(
        self,
        lookback: int = 60,
        hidden_size: int = 128,
        num_layers: int = 3,
        learning_rate: float = 1e-3,
    ):
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = learning_rate
        self.device = get_device()
        self.model: FinancialLSTM | None = None
        self.scaler_params: dict | None = None

    def _prepare_features(self, prices: np.ndarray) -> np.ndarray:
        """Create features from price series: returns, volatility, momentum."""
        n = len(prices)
        features = np.zeros((n, 5))

        features[1:, 0] = np.diff(np.log(prices))

        for i in range(5, n):
            features[i, 1] = np.std(features[i - 5:i, 0])

        for i in range(10, n):
            features[i, 2] = np.mean(features[i - 10:i, 0])

        for i in range(20, n):
            features[i, 3] = np.mean(features[i - 20:i, 0])

        for i in range(1, n):
            features[i, 4] = (prices[i] - prices[max(0, i - 14)]) / prices[max(0, i - 14)]

        return features

    def _create_sequences(
        self, features: np.ndarray, targets: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        xs, ys = [], []
        for i in range(self.lookback, len(features)):
            xs.append(features[i - self.lookback:i])
            ys.append(targets[i])
        return (
            torch.FloatTensor(np.array(xs)).to(self.device),
            torch.FloatTensor(np.array(ys)).to(self.device),
        )

    def train(
        self, prices: np.ndarray, epochs: int = 50, batch_size: int = 32
    ) -> dict:
        """Train LSTM on price history."""
        log.info("lstm.train", epochs=epochs, device=str(self.device), n_prices=len(prices))

        features = self._prepare_features(prices)
        targets = np.diff(np.log(prices))
        features = features[1:]

        self.scaler_params = {
            "mean": features.mean(axis=0).tolist(),
            "std": (features.std(axis=0) + 1e-8).tolist(),
        }
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)

        X, y = self._create_sequences(features, targets)

        self.model = FinancialLSTM(
            input_size=features.shape[1],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        ).to(self.device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.HuberLoss()

        losses = []
        self.model.train()

        for epoch in range(epochs):
            perm = torch.randperm(len(X))
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, len(X), batch_size):
                idx = perm[i:i + batch_size]
                batch_x, batch_y = X[idx], y[idx].unsqueeze(-1)

                optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)

            if (epoch + 1) % 10 == 0:
                log.info("lstm.epoch", epoch=epoch + 1, loss=round(avg_loss, 6))

        return {"final_loss": losses[-1], "epochs": epochs, "device": str(self.device)}

    def predict(self, prices: np.ndarray, horizon: int = 30) -> dict:
        """Generate multi-step forecast."""
        if self.model is None:
            self.train(prices)

        assert self.model is not None

        self.model.eval()
        features = self._prepare_features(prices)
        features = features[1:]

        mean = np.array(self.scaler_params["mean"]) if self.scaler_params else features.mean(axis=0)
        std = np.array(self.scaler_params["std"]) if self.scaler_params else (features.std(axis=0) + 1e-8)
        features = (features - mean) / std

        last_seq = torch.FloatTensor(features[-self.lookback:]).unsqueeze(0).to(self.device)
        predictions = []
        current_price = float(prices[-1])

        with torch.no_grad():
            current_seq = last_seq.clone()
            for _ in range(horizon):
                pred_return = self.model(current_seq).item()
                new_price = current_price * np.exp(pred_return)
                predictions.append(new_price)
                current_price = new_price

                # Build approximate feature vector for the predicted step.
                # Feature 0: log return (from model prediction)
                # Feature 1: 5-day rolling volatility (use recent predicted returns)
                # Feature 2: 10-day momentum (mean of recent returns)
                # Feature 3: 20-day momentum (mean of recent returns)
                # Feature 4: 14-day RSI-style momentum
                new_feature = np.zeros((1, 1, features.shape[1]))
                new_feature[0, 0, 0] = pred_return

                # Gather recent predicted returns for rolling stats
                all_returns = [pred_return]
                if len(predictions) >= 2:
                    for j in range(max(0, len(predictions) - 20), len(predictions) - 1):
                        prev_p = predictions[j]
                        next_p = predictions[j + 1] if j + 1 < len(predictions) else new_price
                        all_returns.append(np.log(next_p / prev_p))

                recent_returns = all_returns[-5:]
                new_feature[0, 0, 1] = np.std(recent_returns) if len(recent_returns) >= 2 else abs(pred_return)

                recent_10 = all_returns[-10:]
                new_feature[0, 0, 2] = np.mean(recent_10)

                recent_20 = all_returns[-20:]
                new_feature[0, 0, 3] = np.mean(recent_20)

                # RSI-style momentum: cumulative return over available window
                if len(predictions) >= 2:
                    lookback_price = predictions[max(0, len(predictions) - 14)]
                    new_feature[0, 0, 4] = (new_price - lookback_price) / lookback_price
                else:
                    new_feature[0, 0, 4] = pred_return

                # Normalize the new feature using saved scaler params
                raw_feat = new_feature[0, 0, :]
                raw_feat = (raw_feat - mean) / std
                new_feature[0, 0, :] = raw_feat

                new_feat_t = torch.FloatTensor(new_feature).to(self.device)
                current_seq = torch.cat([current_seq[:, 1:, :], new_feat_t], dim=1)

        predicted = np.array(predictions)
        returns_std = np.std(np.diff(np.log(prices)))

        # Use log-normal confidence intervals to guarantee positive prices.
        # For a log-normal process: CI = exp(log(pred) +/- z * sigma * sqrt(t))
        days = np.arange(1, horizon + 1)
        z_spread = 1.96 * returns_std * np.sqrt(days)

        return {
            "predicted": predicted,
            "lower": predicted * np.exp(-z_spread),
            "upper": predicted * np.exp(z_spread),
        }
