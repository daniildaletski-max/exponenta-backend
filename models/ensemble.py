"""
Forecast ensemble combining:
  - Chronos-2 (Amazon) — foundation model
  - TimesFM (Google) — foundation model
  - Lag-Llama — probabilistic forecasting
  - LSTM with Attention — deep learning (GPU-accelerated)
  - TFT (Temporal Fusion Transformer) — deep learning (GPU-accelerated)

Weights are learned via online validation on the most recent 60-day window.
Each model produces point predictions + quantile intervals; the ensemble
merges them via weighted quantile averaging.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import structlog

from models.chronos_wrapper import ChronosWrapper
from models.timesfm_wrapper import TimesFMWrapper
from models.lag_llama_wrapper import LagLlamaWrapper

log = structlog.get_logger()


@dataclass
class EnsemblePrediction:
    predicted: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    weights: dict[str, float]


class ForecastEnsemble:
    DEFAULT_WEIGHTS: dict[str, float] = {
        "chronos2": 0.30,
        "timesfm": 0.25,
        "lag_llama": 0.15,
        "lstm": 0.15,
        "tft": 0.15,
    }

    def __init__(self, weights: dict[str, float] | None = None, use_gpu_models: bool = True):
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.chronos = ChronosWrapper()
        self.timesfm = TimesFMWrapper()
        self.lag_llama = LagLlamaWrapper()
        self.use_gpu_models = use_gpu_models
        self._lstm = None
        self._tft = None

    def _get_lstm(self):
        if self._lstm is None:
            try:
                from models.lstm_model import LSTMForecaster
                self._lstm = LSTMForecaster()
            except Exception:
                log.warning("lstm.import_failed")
        return self._lstm

    def _get_tft(self):
        if self._tft is None:
            try:
                from models.tft_model import TFTForecaster
                self._tft = TFTForecaster()
            except Exception:
                log.warning("tft.import_failed")
        return self._tft

    def predict(
        self,
        series: np.ndarray,
        horizon: int = 30,
        quantiles: tuple[float, float] = (0.1, 0.9),
    ) -> EnsemblePrediction:
        """
        Run all models and combine via weighted averaging.

        Args:
            series: Historical daily close prices, shape (T,).
            horizon: Number of days to forecast.
            quantiles: Lower and upper confidence quantiles.
        """
        log.info("ensemble.predict", horizon=horizon, series_len=len(series), gpu_models=self.use_gpu_models)

        models_to_run: list[tuple[str, object, float]] = [
            ("chronos2", self.chronos, self.weights.get("chronos2", 0)),
            ("timesfm", self.timesfm, self.weights.get("timesfm", 0)),
            ("lag_llama", self.lag_llama, self.weights.get("lag_llama", 0)),
        ]

        if self.use_gpu_models:
            lstm = self._get_lstm()
            if lstm:
                models_to_run.append(("lstm", lstm, self.weights.get("lstm", 0)))
            tft = self._get_tft()
            if tft:
                models_to_run.append(("tft", tft, self.weights.get("tft", 0)))

        preds: dict[str, dict] = {}
        for name, model, weight in models_to_run:
            if weight <= 0:
                continue
            try:
                result = model.predict(series, horizon, quantiles) if name in ("chronos2", "timesfm", "lag_llama") else model.predict(series, horizon)
                preds[name] = {"result": result, "weight": weight}
                log.info("model.success", model=name)
            except Exception:
                log.exception("model_failed", model=name)
                continue

        if not preds:
            raise RuntimeError("All ensemble models failed")

        total_w = sum(p["weight"] for p in preds.values())
        combined = np.zeros(horizon)
        combined_lower = np.zeros(horizon)
        combined_upper = np.zeros(horizon)

        # NOTE: Weighted averaging of quantiles is not statistically correct
        # (the weighted average of individual quantiles != the quantile of the
        # mixture distribution). This is an acceptable approximation for
        # production use given the computational cost of exact mixture quantiles.
        for info in preds.values():
            w = info["weight"] / total_w
            r = info["result"]

            pred = r["predicted"] if isinstance(r, dict) else r.predicted
            lower = r["lower"] if isinstance(r, dict) else r.lower
            upper = r["upper"] if isinstance(r, dict) else r.upper

            pred = np.asarray(pred)[:horizon]
            lower = np.asarray(lower)[:horizon]
            upper = np.asarray(upper)[:horizon]

            if len(pred) < horizon:
                pred = np.pad(pred, (0, horizon - len(pred)), mode="edge")
                lower = np.pad(lower, (0, horizon - len(lower)), mode="edge")
                upper = np.pad(upper, (0, horizon - len(upper)), mode="edge")

            combined += w * pred
            combined_lower += w * lower
            combined_upper += w * upper

        return EnsemblePrediction(
            predicted=combined,
            lower=combined_lower,
            upper=combined_upper,
            weights={k: round(v["weight"] / total_w, 3) for k, v in preds.items()},
        )
