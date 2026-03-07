"""
Wrapper for Amazon Chronos-2 time-series foundation model.

Chronos-2 uses a T5-based architecture pre-trained on diverse time-series
corpora. It supports zero-shot forecasting with probabilistic outputs.
"""

from __future__ import annotations

import numpy as np
import structlog

log = structlog.get_logger()


class ChronosWrapper:
    MODEL_ID = "amazon/chronos-t5-large"

    def __init__(self):
        self._pipeline = None

    def _load(self):
        if self._pipeline is not None:
            return
        try:
            from chronos import ChronosPipeline
            self._pipeline = ChronosPipeline.from_pretrained(
                self.MODEL_ID,
                device_map="auto",
            )
            log.info("chronos.loaded", model=self.MODEL_ID)
        except ImportError:
            log.warning("chronos.not_installed — using stub predictions")

    def predict(
        self,
        series: np.ndarray,
        horizon: int = 30,
        quantiles: tuple[float, float] = (0.1, 0.9),
    ) -> dict[str, np.ndarray]:
        self._load()

        if self._pipeline is not None:
            import torch
            context = torch.tensor(series, dtype=torch.float32).unsqueeze(0)
            forecast = self._pipeline.predict(context, prediction_length=horizon, num_samples=100)
            samples = forecast.detach().cpu().numpy()[0]
            return {
                "predicted": np.median(samples, axis=0),
                "lower": np.quantile(samples, quantiles[0], axis=0),
                "upper": np.quantile(samples, quantiles[1], axis=0),
            }

        # Stub: random walk with drift
        last = series[-1]
        drift = np.mean(np.diff(series[-60:])) if len(series) > 60 else 0
        noise = np.random.default_rng(42).normal(0, last * 0.01, horizon)
        predicted = last + np.cumsum(np.full(horizon, drift) + noise)
        return {
            "predicted": predicted,
            "lower": predicted * 0.95,
            "upper": predicted * 1.05,
        }
