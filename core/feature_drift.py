"""
Feature Drift Detection -- monitors data distribution shifts that degrade model accuracy.

Implements:
- Population Stability Index (PSI) for distribution comparison
- Feature importance tracking over time
- Drift alerts when distributions shift beyond thresholds
- Automated feature health scoring
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np


@dataclass
class DriftReport:
    feature_name: str
    psi: float
    drift_detected: bool
    reference_mean: float
    current_mean: float
    reference_std: float
    current_std: float
    severity: str  # "none" | "minor" | "major" | "critical"


@dataclass
class FeatureHealth:
    feature_name: str
    importance: float
    drift_psi: float
    health_score: float  # 0-1, lower = worse
    status: str  # "healthy" | "degraded" | "critical"


class FeatureDriftDetector:
    """Monitors feature distributions for drift that may degrade predictions."""

    # PSI thresholds
    PSI_MINOR = 0.1
    PSI_MAJOR = 0.2
    PSI_CRITICAL = 0.5

    def compute_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Compute Population Stability Index between two distributions.

        PSI < 0.1  : no significant shift
        PSI 0.1-0.2: moderate shift (monitor)
        PSI > 0.2  : significant shift (retrain)
        """
        reference = np.asarray(reference, dtype=np.float64)
        current = np.asarray(current, dtype=np.float64)

        # Create bins from reference distribution
        bins = np.linspace(
            min(reference.min(), current.min()),
            max(reference.max(), current.max()),
            n_bins + 1,
        )

        ref_counts, _ = np.histogram(reference, bins=bins)
        cur_counts, _ = np.histogram(current, bins=bins)

        # Normalize to proportions (add small epsilon to avoid log(0))
        eps = 1e-8
        ref_pct = (ref_counts + eps) / (len(reference) + eps * n_bins)
        cur_pct = (cur_counts + eps) / (len(current) + eps * n_bins)

        # PSI formula
        psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
        return round(max(psi, 0.0), 6)

    def classify_drift(self, psi: float) -> str:
        """Classify drift severity based on PSI value."""
        if psi < self.PSI_MINOR:
            return "none"
        elif psi < self.PSI_MAJOR:
            return "minor"
        elif psi < self.PSI_CRITICAL:
            return "major"
        return "critical"

    def analyze_feature(
        self,
        feature_name: str,
        reference: np.ndarray,
        current: np.ndarray,
    ) -> DriftReport:
        """Analyze a single feature for distribution drift."""
        reference = np.asarray(reference, dtype=np.float64)
        current = np.asarray(current, dtype=np.float64)

        psi = self.compute_psi(reference, current)
        severity = self.classify_drift(psi)

        return DriftReport(
            feature_name=feature_name,
            psi=psi,
            drift_detected=psi >= self.PSI_MINOR,
            reference_mean=float(np.mean(reference)),
            current_mean=float(np.mean(current)),
            reference_std=float(np.std(reference)),
            current_std=float(np.std(current)),
            severity=severity,
        )

    def analyze_all_features(
        self,
        feature_names: list[str],
        reference_matrix: np.ndarray,
        current_matrix: np.ndarray,
        importances: np.ndarray | None = None,
    ) -> list[FeatureHealth]:
        """Analyze all features and compute health scores.

        Parameters
        ----------
        feature_names : list[str]
            Names of features.
        reference_matrix : np.ndarray
            (T_ref, N_features) reference period data.
        current_matrix : np.ndarray
            (T_cur, N_features) current period data.
        importances : np.ndarray | None
            Feature importance scores (0-1). If None, equal importance.

        Returns
        -------
        list[FeatureHealth] sorted by health_score ascending (worst first).
        """
        reference_matrix = np.asarray(reference_matrix, dtype=np.float64)
        current_matrix = np.asarray(current_matrix, dtype=np.float64)
        n_features = len(feature_names)

        if importances is None:
            importances = np.ones(n_features) / n_features
        else:
            importances = np.asarray(importances, dtype=np.float64)

        results = []
        for i, name in enumerate(feature_names):
            psi = self.compute_psi(reference_matrix[:, i], current_matrix[:, i])
            severity = self.classify_drift(psi)

            # Health score: high importance + high drift = low health
            drift_penalty = min(psi / self.PSI_CRITICAL, 1.0)
            health = max(0.0, 1.0 - drift_penalty * importances[i] * 2)

            if health > 0.7:
                status = "healthy"
            elif health > 0.3:
                status = "degraded"
            else:
                status = "critical"

            results.append(FeatureHealth(
                feature_name=name,
                importance=round(float(importances[i]), 4),
                drift_psi=psi,
                health_score=round(health, 4),
                status=status,
            ))

        return sorted(results, key=lambda x: x.health_score)

    def get_retrain_recommendation(self, health_reports: list[FeatureHealth]) -> dict:
        """Determine if model should be retrained based on feature health."""
        if not health_reports:
            return {"retrain": False, "reason": "No features to analyze"}

        critical = [h for h in health_reports if h.status == "critical"]
        degraded = [h for h in health_reports if h.status == "degraded"]
        avg_health = sum(h.health_score for h in health_reports) / len(health_reports)

        if critical:
            return {
                "retrain": True,
                "urgency": "immediate",
                "reason": f"{len(critical)} critical features: {', '.join(h.feature_name for h in critical)}",
                "avg_health": round(avg_health, 4),
            }
        elif degraded and avg_health < 0.6:
            return {
                "retrain": True,
                "urgency": "soon",
                "reason": f"{len(degraded)} degraded features, avg health {avg_health:.2f}",
                "avg_health": round(avg_health, 4),
            }
        return {
            "retrain": False,
            "urgency": "none",
            "reason": f"All features healthy, avg health {avg_health:.2f}",
            "avg_health": round(avg_health, 4),
        }


# Singleton
drift_detector = FeatureDriftDetector()
