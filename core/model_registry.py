"""
Model Registry -- tracks ML model versions, metrics, and lifecycle.

Supports:
- Version tracking for each model (chronos, timesfm, lagllama, ensemble, etc.)
- Performance metrics per version
- Active/inactive model management
- Automated comparison of model versions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class ModelInfo:
    model_name: str
    version: int
    metrics: dict[str, float] = field(default_factory=dict)
    training_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_active: bool = True
    file_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelRegistry:
    """In-memory model registry for beta. Replaced by DB in production."""

    def __init__(self):
        self._models: dict[str, list[ModelInfo]] = {}
        self._seed_defaults()

    def _seed_defaults(self):
        """Seed registry with current model versions."""
        defaults = [
            ("chronos_t5", {"mae": 0.023, "mape": 2.8, "directional_accuracy": 0.61}),
            ("timesfm", {"mae": 0.019, "mape": 2.3, "directional_accuracy": 0.64}),
            ("lag_llama", {"mae": 0.026, "mape": 3.1, "directional_accuracy": 0.58}),
            ("ensemble_v1", {"mae": 0.018, "mape": 2.1, "directional_accuracy": 0.67}),
            ("xgboost_signal", {"precision": 0.72, "recall": 0.68, "f1": 0.70}),
            ("regime_detector", {"accuracy": 0.75, "transition_recall": 0.62}),
        ]
        for name, metrics in defaults:
            self.register(name, version=1, metrics=metrics)

    def register(
        self,
        model_name: str,
        version: int,
        metrics: dict[str, float] | None = None,
        file_path: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ModelInfo:
        """Register a new model version."""
        info = ModelInfo(
            model_name=model_name,
            version=version,
            metrics=metrics or {},
            file_path=file_path,
            metadata=metadata or {},
        )
        if model_name not in self._models:
            self._models[model_name] = []
        self._models[model_name].append(info)
        return info

    def get_active(self, model_name: str) -> ModelInfo | None:
        """Get the active version of a model."""
        versions = self._models.get(model_name, [])
        for v in reversed(versions):
            if v.is_active:
                return v
        return None

    def get_all_versions(self, model_name: str) -> list[ModelInfo]:
        """Get all versions of a model."""
        return list(self._models.get(model_name, []))

    def list_models(self) -> dict[str, ModelInfo | None]:
        """List all registered models with their active version."""
        return {name: self.get_active(name) for name in self._models}

    def promote(self, model_name: str, version: int) -> bool:
        """Promote a specific version to active, deactivating others."""
        versions = self._models.get(model_name, [])
        found = False
        for v in versions:
            if v.version == version:
                v.is_active = True
                found = True
            else:
                v.is_active = False
        return found

    def compare_versions(self, model_name: str, metric: str = "mae") -> list[dict]:
        """Compare all versions by a specific metric."""
        versions = self._models.get(model_name, [])
        results = []
        for v in versions:
            results.append({
                "version": v.version,
                "is_active": v.is_active,
                metric: v.metrics.get(metric),
                "training_date": v.training_date.isoformat(),
            })
        return sorted(results, key=lambda x: x.get(metric) or float("inf"))

    def should_retrain(self, model_name: str, max_age_days: int = 7) -> bool:
        """Check if a model needs retraining based on age."""
        active = self.get_active(model_name)
        if not active:
            return True
        age = (datetime.now(timezone.utc) - active.training_date).days
        return age >= max_age_days

    def get_stale_models(self, max_age_days: int = 7) -> list[str]:
        """Get list of models that need retraining."""
        return [
            name for name in self._models
            if self.should_retrain(name, max_age_days)
        ]


# Singleton
registry = ModelRegistry()
