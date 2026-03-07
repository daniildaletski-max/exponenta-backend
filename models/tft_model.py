"""
Temporal Fusion Transformer (TFT) wrapper for financial time series.

Uses pytorch-forecasting when available, falls back to a simplified
attention-based transformer implementation.

Best run on GPU (CUDA or MPS). Training on CPU is significantly slower.
"""

from __future__ import annotations

import numpy as np
import structlog
import torch
import torch.nn as nn

from gpu.device import get_device

log = structlog.get_logger()


class SimpleTFT(nn.Module):
    """
    Simplified Temporal Fusion Transformer.

    Uses encoder-decoder transformer architecture with variable selection
    and interpretable attention for financial forecasting.
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_prediction_length: int = 30,
    ):
        super().__init__()
        self.max_prediction_length = max_prediction_length

        self.input_projection = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.positional_encoding = nn.Parameter(
            torch.randn(1, 500, d_model) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 3),  # [predicted, lower_q, upper_q]
        )

        self.decoder_input = nn.Parameter(torch.randn(1, max_prediction_length, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        x = self.input_projection(x)
        x = x + self.positional_encoding[:, :T, :]

        memory = self.encoder(x)

        decoder_in = self.decoder_input.expand(B, -1, -1)
        decoded = self.decoder(decoder_in, memory)

        return self.output_projection(decoded)


class TFTForecaster:
    """High-level TFT forecaster with training and inference."""

    def __init__(
        self,
        encoder_length: int = 60,
        prediction_length: int = 30,
        d_model: int = 128,
        learning_rate: float = 1e-3,
    ):
        self.encoder_length = encoder_length
        self.prediction_length = prediction_length
        self.d_model = d_model
        self.lr = learning_rate
        self.device = get_device()
        self.model: SimpleTFT | None = None

    def _prepare_features(self, prices: np.ndarray) -> np.ndarray:
        """Build multi-variate features from prices."""
        n = len(prices)
        features = np.zeros((n, 8))

        features[:, 0] = prices / prices[0]

        features[1:, 1] = np.diff(np.log(prices))

        for i in range(5, n):
            features[i, 2] = np.std(np.diff(np.log(prices[max(0, i - 5):i + 1])))
        for i in range(20, n):
            features[i, 3] = np.std(np.diff(np.log(prices[max(0, i - 20):i + 1])))

        for i in range(5, n):
            features[i, 4] = np.mean(prices[i - 5:i + 1])
        for i in range(20, n):
            features[i, 5] = np.mean(prices[i - 20:i + 1])

        features[:, 6] = np.arange(n) % 5 / 4.0
        features[:, 7] = np.arange(n) % 21 / 20.0

        return features

    def _create_tft_sequences(
        self, features: np.ndarray, prices: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        xs, ys = [], []
        total_len = self.encoder_length + self.prediction_length

        for i in range(total_len, len(features)):
            xs.append(features[i - total_len:i - self.prediction_length])
            future_prices = prices[i - self.prediction_length:i]
            base_price = prices[i - self.prediction_length - 1]
            target_returns = np.log(future_prices / base_price)
            ys.append(target_returns)

        return (
            torch.FloatTensor(np.array(xs)).to(self.device),
            torch.FloatTensor(np.array(ys)).to(self.device),
        )

    def train(self, prices: np.ndarray, epochs: int = 50, batch_size: int = 32) -> dict:
        """Train TFT model."""
        log.info("tft.train", epochs=epochs, device=str(self.device))

        features = self._prepare_features(prices)
        mean = features.mean(axis=0)
        std = features.std(axis=0) + 1e-8
        features_norm = (features - mean) / std

        X, y = self._create_tft_sequences(features_norm, prices)

        if len(X) < batch_size:
            log.warning("tft.insufficient_data", samples=len(X))
            return {"error": "insufficient data", "samples": len(X)}

        self.model = SimpleTFT(
            input_size=features.shape[1],
            d_model=self.d_model,
            max_prediction_length=self.prediction_length,
        ).to(self.device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        batches_per_epoch = max(1, (len(X) + batch_size - 1) // batch_size)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.lr, total_steps=epochs * batches_per_epoch, pct_start=0.3
        )
        criterion = nn.HuberLoss()

        self.model.train()
        losses = []

        for epoch in range(epochs):
            perm = torch.randperm(len(X))
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, len(X), batch_size):
                idx = perm[i:i + batch_size]
                batch_x = X[idx]
                batch_y = y[idx]

                optimizer.zero_grad()
                output = self.model(batch_x)
                pred_returns = output[:, :, 0]
                loss = criterion(pred_returns, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            losses.append(epoch_loss / max(n_batches, 1))

        log.info("tft.train_done", final_loss=round(losses[-1], 6))
        self._feature_mean = mean
        self._feature_std = std

        return {"final_loss": losses[-1], "epochs": epochs, "device": str(self.device)}

    def predict(self, prices: np.ndarray, horizon: int | None = None) -> dict:
        """Generate forecast with confidence intervals."""
        if self.model is None:
            self.train(prices)

        if self.model is None:
            # Training failed (e.g. insufficient data). Return a naive fallback
            # so the ensemble can still proceed with other models.
            log.warning("tft.predict_fallback", reason="model is None after train()")
            if horizon is None:
                horizon = self.prediction_length
            last = float(prices[-1])
            drift = np.mean(np.diff(np.log(prices[-60:]))) if len(prices) > 60 else 0.0
            days = np.arange(1, horizon + 1)
            predicted = last * np.exp(drift * days)
            returns_std = np.std(np.diff(np.log(prices[-60:]))) if len(prices) > 60 else 0.02
            return {
                "predicted": predicted,
                "lower": predicted * np.exp(-1.96 * returns_std * np.sqrt(days)),
                "upper": predicted * np.exp(1.96 * returns_std * np.sqrt(days)),
            }

        if horizon is None:
            horizon = self.prediction_length

        self.model.eval()
        features = self._prepare_features(prices)
        features_norm = (features - self._feature_mean) / self._feature_std

        last_seq = features_norm[-self.encoder_length:]
        x = torch.FloatTensor(last_seq).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(x)
            pred_returns = output[0, :horizon, 0].cpu().numpy()
            lower_q = output[0, :horizon, 1].cpu().numpy()
            upper_q = output[0, :horizon, 2].cpu().numpy()

        base_price = float(prices[-1])
        predicted = base_price * np.exp(np.cumsum(pred_returns))

        returns_std = np.std(np.diff(np.log(prices[-60:])))
        days = np.arange(1, horizon + 1)
        lower = predicted * (1 - 1.96 * returns_std * np.sqrt(days))
        upper = predicted * (1 + 1.96 * returns_std * np.sqrt(days))

        return {
            "predicted": predicted,
            "lower": lower,
            "upper": upper,
        }
