import logging
import os
import threading
import time
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import joblib
import optuna
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

from prediction.features import (
    build_feature_matrix, get_feature_cols, fetch_ohlcv, compute_features, FORWARD_DAYS,
    DEFAULT_LOOKBACK
)

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

OPTUNA_TRIALS = 18
MIN_TRAIN_SIZE = 200
RANDOM_STATE = 42
CACHE_TTL_PREDICTION = 300

WALK_FORWARD_FOLDS = 5
PURGE_GAP = 10
EMBARGO_GAP = 5

STALENESS_ACCURACY_DROP = 0.05
STALENESS_MAX_AGE_HOURS = 24

MULTI_HORIZONS = [1, 5, 20]


class ProductionEnsemble:

    def __init__(self, n_trials: int = OPTUNA_TRIALS):
        self.n_trials = n_trials
        self.models = {}
        self.horizon_models: Dict[int, Dict[str, Any]] = {}
        self.scaler = StandardScaler()
        self.horizon_scalers: Dict[int, StandardScaler] = {}
        self.feature_cols = []
        self.important_features = []
        self.feature_importance_ranked = []
        self.feature_interactions: List[Dict[str, Any]] = []
        self.fundamental_feature_importance = {}
        self.is_fitted = False
        self.best_params = {}
        self.cv_accuracy = 0.0
        self.wf_accuracy = 0.0
        self.model_weights = [0.34, 0.33, 0.33]
        self.regime = "neutral"
        self.regime_weights = {"momentum": 0.5, "value": 0.5}
        self.conformal_scores: Optional[np.ndarray] = None
        self.conformal_quantiles: Dict[str, float] = {}
        self.horizon_accuracies: Dict[int, float] = {}
        self.fitted_at: float = 0.0
        self.staleness_info: Dict[str, Any] = {}

    def _prepare(self, df: pd.DataFrame, target_col: str = "fwd_dir_5d") -> Tuple[np.ndarray, np.ndarray]:
        self.feature_cols = get_feature_cols(df)
        X = df[self.feature_cols].copy()
        y = df[target_col].copy()
        valid = y.notna() & np.isfinite(y)
        X = X[valid]
        y = y[valid]
        X = X.ffill().fillna(0)
        X = X.replace([np.inf, -np.inf], 0)

        null_pct = X.isnull().sum() / len(X)
        keep_cols = null_pct[null_pct < 0.5].index.tolist()
        low_var_cols = [c for c in keep_cols if X[c].std() > 1e-8]
        self.feature_cols = low_var_cols
        X = X[self.feature_cols].fillna(0)

        return X.values.astype(np.float32), y.values.astype(int)

    def _detect_regime(self, df: pd.DataFrame) -> str:
        try:
            close = df["close"]
            sma_50 = close.rolling(50).mean()
            sma_200 = close.rolling(200).mean()
            ret_60d = close.pct_change(60).iloc[-1] if len(close) > 60 else 0
            vol_20d = close.pct_change().tail(20).std()
            vol_60d = close.pct_change().tail(60).std()

            above_sma50 = close.iloc[-1] > sma_50.iloc[-1] if not pd.isna(sma_50.iloc[-1]) else True
            above_sma200 = close.iloc[-1] > sma_200.iloc[-1] if not pd.isna(sma_200.iloc[-1]) else True
            sma50_above_200 = sma_50.iloc[-1] > sma_200.iloc[-1] if not pd.isna(sma_200.iloc[-1]) else True
            vol_expanding = vol_20d > vol_60d * 1.5 if vol_60d > 0 else False

            bullish_signals = sum([above_sma50, above_sma200, sma50_above_200, ret_60d > 0.05])

            # Crisis: sharp drawdown + high vol + below both MAs
            if ret_60d < -0.15 and vol_expanding and not above_sma200:
                self.regime = "crisis"
                self.regime_weights = {"momentum": 0.2, "value": 0.8}
            # Volatile: expanding volatility but no clear trend
            elif vol_expanding and 1 <= bullish_signals <= 2:
                self.regime = "volatile"
                self.regime_weights = {"momentum": 0.4, "value": 0.6}
            elif bullish_signals >= 3:
                self.regime = "bullish"
                self.regime_weights = {"momentum": 0.65, "value": 0.35}
            elif bullish_signals <= 1:
                self.regime = "bearish"
                self.regime_weights = {"momentum": 0.35, "value": 0.65}
            else:
                self.regime = "neutral"
                self.regime_weights = {"momentum": 0.5, "value": 0.5}
        except Exception:
            self.regime = "neutral"
            self.regime_weights = {"momentum": 0.5, "value": 0.5}

        return self.regime

    def _purged_walk_forward_validate(self, X: np.ndarray, y: np.ndarray, params: dict) -> float:
        n = len(X)
        fold_size = n // (WALK_FORWARD_FOLDS + 1)
        if fold_size < 50:
            return 0.0

        accuracies = []
        for fold in range(WALK_FORWARD_FOLDS):
            train_end = fold_size * (fold + 1)
            purge_end = min(train_end + PURGE_GAP, n)
            test_start = purge_end + EMBARGO_GAP
            test_end = min(test_start + fold_size, n)

            if test_end <= test_start or test_start >= n:
                continue

            X_tr, y_tr = X[:train_end], y[:train_end]
            X_te, y_te = X[test_start:test_end], y[test_start:test_end]

            if len(X_te) < 10:
                continue

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            model = xgb.XGBClassifier(**params, random_state=RANDOM_STATE, verbosity=0, use_label_encoder=False)
            model.fit(X_tr_s, y_tr)
            pred = model.predict(X_te_s)
            accuracies.append(accuracy_score(y_te, pred))

        return float(np.mean(accuracies)) if accuracies else 0.0

    def _compute_conformal_scores(self, X_cal: np.ndarray, y_cal: np.ndarray) -> None:
        try:
            probs = []
            for name, model in self.models.items():
                try:
                    p = model.predict_proba(X_cal)[:, 1]
                    probs.append(p)
                except Exception:
                    pass

            if not probs:
                return

            weights = self.model_weights[:len(probs)]
            w_sum = sum(weights)
            weights = [w / w_sum for w in weights] if w_sum > 0 else [1 / len(probs)] * len(probs)

            avg_probs = np.average(probs, axis=0, weights=weights)

            scores = np.where(y_cal == 1, 1 - avg_probs, avg_probs)
            self.conformal_scores = np.sort(scores)

            n_cal = len(self.conformal_scores)
            for alpha, label in [(0.1, "90"), (0.05, "95"), (0.01, "99")]:
                q_idx = int(np.ceil((1 - alpha) * (n_cal + 1))) - 1
                q_idx = min(q_idx, n_cal - 1)
                self.conformal_quantiles[label] = float(self.conformal_scores[q_idx])

            logger.debug(f"Conformal quantiles: {self.conformal_quantiles}")
        except Exception as e:
            logger.debug(f"Conformal prediction setup failed: {e}")

    def _detect_feature_interactions(self, X: np.ndarray, y: np.ndarray) -> None:
        try:
            n_features = X.shape[1]
            if n_features < 2 or len(X) < 100:
                return

            top_k = min(20, n_features)
            if hasattr(self, 'feature_importance_ranked') and self.feature_importance_ranked:
                top_names = [f["feature"] for f in self.feature_importance_ranked[:top_k]]
                top_indices = [self.feature_cols.index(n) for n in top_names if n in self.feature_cols]
            else:
                variances = np.var(X, axis=0)
                top_indices = np.argsort(variances)[::-1][:top_k].tolist()

            interactions = []
            for i in range(len(top_indices)):
                for j in range(i + 1, len(top_indices)):
                    idx_a, idx_b = top_indices[i], top_indices[j]
                    interaction = X[:, idx_a] * X[:, idx_b]

                    if np.std(interaction) < 1e-10:
                        continue

                    corr = np.corrcoef(interaction, y)[0, 1]
                    if np.isnan(corr):
                        continue

                    corr_a = abs(np.corrcoef(X[:, idx_a], y)[0, 1]) if not np.isnan(np.corrcoef(X[:, idx_a], y)[0, 1]) else 0
                    corr_b = abs(np.corrcoef(X[:, idx_b], y)[0, 1]) if not np.isnan(np.corrcoef(X[:, idx_b], y)[0, 1]) else 0
                    synergy = abs(corr) - max(corr_a, corr_b)

                    if synergy > 0.01:
                        interactions.append({
                            "feature_a": self.feature_cols[idx_a] if idx_a < len(self.feature_cols) else f"f{idx_a}",
                            "feature_b": self.feature_cols[idx_b] if idx_b < len(self.feature_cols) else f"f{idx_b}",
                            "interaction_corr": round(float(abs(corr)), 4),
                            "synergy": round(float(synergy), 4),
                            "direction": "positive" if corr > 0 else "negative",
                        })

            interactions.sort(key=lambda x: x["synergy"], reverse=True)
            self.feature_interactions = interactions[:10]
        except Exception as e:
            logger.debug(f"Feature interaction detection failed: {e}")

    def _compute_feature_importance(self, xgb_model, lgb_model) -> None:
        try:
            xgb_imp = xgb_model.feature_importances_
            lgb_imp = lgb_model.feature_importances_

            if len(xgb_imp) == len(self.feature_cols) and len(lgb_imp) == len(self.feature_cols):
                xgb_norm = xgb_imp / (xgb_imp.sum() + 1e-10)
                lgb_norm = lgb_imp / (lgb_imp.sum() + 1e-10)
                combined = (xgb_norm + lgb_norm) / 2

                ranked = sorted(
                    zip(self.feature_cols, combined),
                    key=lambda x: x[1],
                    reverse=True
                )
                self.feature_importance_ranked = [
                    {"feature": name, "importance": round(float(imp), 6)}
                    for name, imp in ranked[:30]
                ]

                importance_mask = combined > np.percentile(combined, 10)
                self.important_features = [
                    self.feature_cols[i] for i in range(len(self.feature_cols)) if importance_mask[i]
                ]

                fund_features = {}
                for i, name in enumerate(self.feature_cols):
                    if name.startswith("fund_") or name.startswith("insider_") or name.startswith("institutional_"):
                        fund_features[name] = round(float(combined[i]), 6)
                self.fundamental_feature_importance = dict(
                    sorted(fund_features.items(), key=lambda x: x[1], reverse=True)
                )
            else:
                self.important_features = self.feature_cols
        except Exception:
            self.important_features = self.feature_cols

    def _train_horizon_model(self, df: pd.DataFrame, horizon: int) -> Dict[str, Any]:
        target_col = f"fwd_dir_{horizon}d"
        if target_col not in df.columns:
            return {}

        feature_cols = get_feature_cols(df)
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        valid = y.notna() & np.isfinite(y)
        X = X[valid]
        y = y[valid]
        X = X.ffill().fillna(0).replace([np.inf, -np.inf], 0)

        null_pct = X.isnull().sum() / len(X)
        keep_cols = null_pct[null_pct < 0.5].index.tolist()
        low_var_cols = [c for c in keep_cols if X[c].std() > 1e-8]
        X = X[low_var_cols].fillna(0)

        if len(X) < MIN_TRAIN_SIZE:
            return {}

        X_arr = X.values.astype(np.float32)
        y_arr = y.values.astype(int)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_arr)

        train_end = int(len(X_arr) * 0.8)
        X_train, y_train = X_scaled[:train_end], y_arr[:train_end]
        X_test, y_test = X_scaled[train_end:], y_arr[train_end:]

        model = xgb.XGBClassifier(
            n_estimators=self.best_params.get("n_estimators", 200),
            max_depth=self.best_params.get("max_depth", 6),
            learning_rate=self.best_params.get("learning_rate", 0.1),
            subsample=self.best_params.get("subsample", 0.8),
            colsample_bytree=self.best_params.get("colsample_bytree", 0.8),
            random_state=RANDOM_STATE, verbosity=0, use_label_encoder=False
        )
        model.fit(X_train, y_train)

        acc = accuracy_score(y_test, model.predict(X_test))
        self.horizon_models[horizon] = {"model": model, "feature_cols": low_var_cols}
        self.horizon_scalers[horizon] = scaler
        self.horizon_accuracies[horizon] = acc

        return {"horizon": horizon, "accuracy": round(acc, 4)}

    def fit(self, df: pd.DataFrame) -> Dict[str, Any]:
        self._detect_regime(df)

        X, y = self._prepare(df)
        if len(X) < MIN_TRAIN_SIZE:
            return {"error": "Insufficient data", "samples": len(X)}

        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        train_end = int(len(X) * 0.7)
        cal_end = int(len(X) * 0.85)
        X_train, y_train = X_scaled[:train_end], y[:train_end]
        X_cal, y_cal = X_scaled[train_end:cal_end], y[train_end:cal_end]
        X_test, y_test = X_scaled[cal_end:], y[cal_end:]

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 400),
                "max_depth": trial.suggest_int("max_depth", 3, 9),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0, 5.0),
            }
            model = xgb.XGBClassifier(**params, random_state=RANDOM_STATE, verbosity=0, use_label_encoder=False)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            return accuracy_score(y_test, pred)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        self.best_params = study.best_params

        xgb_params = {k: v for k, v in self.best_params.items()}
        xgb_model = xgb.XGBClassifier(
            **xgb_params, random_state=RANDOM_STATE, verbosity=0, use_label_encoder=False
        )

        lgb_model = lgb.LGBMClassifier(
            n_estimators=self.best_params.get("n_estimators", 200),
            max_depth=self.best_params.get("max_depth", 6),
            learning_rate=self.best_params.get("learning_rate", 0.1),
            subsample=self.best_params.get("subsample", 0.8),
            colsample_bytree=self.best_params.get("colsample_bytree", 0.8),
            reg_alpha=self.best_params.get("reg_alpha", 0.01),
            reg_lambda=self.best_params.get("reg_lambda", 0.01),
            min_child_samples=max(5, self.best_params.get("min_child_weight", 5)),
            random_state=RANDOM_STATE, verbose=-1,
        )

        cat_model = CatBoostClassifier(
            iterations=self.best_params.get("n_estimators", 200),
            depth=min(self.best_params.get("max_depth", 6), 10),
            learning_rate=self.best_params.get("learning_rate", 0.1),
            l2_leaf_reg=self.best_params.get("reg_lambda", 1.0),
            random_seed=RANDOM_STATE,
            verbose=0,
        )

        X_train_full = X_scaled[:cal_end]
        y_train_full = y[:cal_end]
        xgb_model.fit(X_train_full, y_train_full)
        lgb_model.fit(X_train_full, y_train_full)
        cat_model.fit(X_train_full, y_train_full)

        self.models = {"xgb": xgb_model, "lgb": lgb_model, "cat": cat_model}
        self.is_fitted = True
        self.fitted_at = time.time()

        self._compute_feature_importance(xgb_model, lgb_model)

        self._compute_conformal_scores(X_test, y_test)

        self._detect_feature_interactions(X_scaled, y)

        xgb_acc = accuracy_score(y_test, xgb_model.predict(X_test))
        lgb_acc = accuracy_score(y_test, lgb_model.predict(X_test))
        cat_acc = accuracy_score(y_test, cat_model.predict(X_test))

        # ── Bayesian weight auto-tuning with regime awareness ──
        MIN_WEIGHT = 0.05
        accs = {"xgb": xgb_acc, "lgb": lgb_acc, "cat": cat_acc}

        regime_bonus = {"xgb": 0.0, "lgb": 0.0, "cat": 0.0}
        if self.regime == "bullish":
            regime_bonus["xgb"] = 0.06
            regime_bonus["lgb"] = 0.03
        elif self.regime == "bearish":
            regime_bonus["cat"] = 0.06
            regime_bonus["lgb"] = 0.03
        elif self.regime == "crisis":
            regime_bonus["cat"] = 0.08
            regime_bonus["xgb"] = -0.02
        elif self.regime == "volatile":
            regime_bonus["lgb"] = 0.05

        adjusted = {k: max(MIN_WEIGHT, accs[k] + regime_bonus.get(k, 0)) for k in accs}
        total = sum(adjusted.values())
        if total > 0:
            self.model_weights = [adjusted["xgb"] / total, adjusted["lgb"] / total, adjusted["cat"] / total]
        else:
            self.model_weights = [0.34, 0.33, 0.33]

        for i in range(3):
            if self.model_weights[i] < MIN_WEIGHT:
                deficit = MIN_WEIGHT - self.model_weights[i]
                self.model_weights[i] = MIN_WEIGHT
                others = [j for j in range(3) if j != i]
                for j in others:
                    self.model_weights[j] -= deficit / len(others)

        logger.info(
            f"Auto-tuned weights: XGB={self.model_weights[0]:.3f} "
            f"LGB={self.model_weights[1]:.3f} CAT={self.model_weights[2]:.3f} "
            f"regime={self.regime}"
        )

        ensemble_pred = (
            self.model_weights[0] * xgb_model.predict_proba(X_test)[:, 1] +
            self.model_weights[1] * lgb_model.predict_proba(X_test)[:, 1] +
            self.model_weights[2] * cat_model.predict_proba(X_test)[:, 1]
        )
        self.cv_accuracy = accuracy_score(y_test, (ensemble_pred >= 0.5).astype(int))

        try:
            self.wf_accuracy = self._purged_walk_forward_validate(X, y, xgb_params)
        except Exception:
            self.wf_accuracy = self.cv_accuracy

        horizon_results = {}
        for h in MULTI_HORIZONS:
            if h == 5:
                self.horizon_accuracies[5] = self.cv_accuracy
                continue
            try:
                hr = self._train_horizon_model(df, h)
                if hr:
                    horizon_results[f"{h}d"] = hr
            except Exception as e:
                logger.debug(f"Horizon {h}d training failed: {e}")

        logger.info(
            f"Ensemble trained: accuracy={self.cv_accuracy:.3f}, wf_accuracy={self.wf_accuracy:.3f}, "
            f"regime={self.regime}, "
            f"XGB:{xgb_acc:.3f} LGB:{lgb_acc:.3f} CAT:{cat_acc:.3f}, "
            f"features={len(self.feature_cols)} (fundamental: {len(self.fundamental_feature_importance)}), "
            f"horizons={list(self.horizon_accuracies.keys())}, "
            f"interactions={len(self.feature_interactions)}"
        )
        return {
            "cv_accuracy": round(self.cv_accuracy, 4),
            "wf_accuracy": round(self.wf_accuracy, 4),
            "regime": self.regime,
            "regime_weights": self.regime_weights,
            "best_params": self.best_params,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "calibration_samples": len(X_cal),
            "feature_count": len(self.feature_cols),
            "important_feature_count": len(self.important_features),
            "fundamental_features_count": len(self.fundamental_feature_importance),
            "top_features": self.feature_importance_ranked[:10],
            "fundamental_feature_importance": self.fundamental_feature_importance,
            "feature_interactions": self.feature_interactions[:5],
            "conformal_quantiles": self.conformal_quantiles,
            "horizon_accuracies": {f"{k}d": round(v, 4) for k, v in self.horizon_accuracies.items()},
            "individual_accuracy": {
                "xgboost": round(xgb_acc, 4),
                "lightgbm": round(lgb_acc, 4),
                "catboost": round(cat_acc, 4),
            },
        }

    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        if not self.is_fitted:
            return {"direction": 0, "probability": 0.5, "confidence": 0.0}

        X_scaled = self.scaler.transform(X.reshape(1, -1) if X.ndim == 1 else X)
        probs = []
        for name, model in self.models.items():
            try:
                p = model.predict_proba(X_scaled)[:, 1]
                probs.append(p)
            except Exception:
                pass

        if not probs:
            return {"direction": 0, "probability": 0.5, "confidence": 0.0}

        weights = self.model_weights[:len(probs)]
        w_sum = sum(weights)
        weights = [w / w_sum for w in weights] if w_sum > 0 else [1 / len(probs)] * len(probs)

        avg_prob = float(np.average(probs, axis=0, weights=weights)[0])
        std_prob = float(np.std([p[0] for p in probs])) if len(probs) > 1 else 0.1
        agreement = sum(1 for p in probs if (p[0] >= 0.5) == (avg_prob >= 0.5)) / len(probs)
        confidence = float(np.clip(agreement * (1.0 - std_prob * 2), 0.1, 0.95))
        direction = int(avg_prob >= 0.5)

        conformal_intervals = self._compute_conformal_intervals(avg_prob)

        shap_factors = self._get_shap(X_scaled)

        return {
            "direction": direction,
            "probability": round(avg_prob, 4),
            "confidence": round(confidence * 100, 1),
            "model_agreement": round(agreement, 3),
            "ensemble_std": round(std_prob, 4),
            "conformal_intervals": conformal_intervals,
            "shap_factors": shap_factors,
            "model_probas": {
                name: round(float(probs[i][0]), 4)
                for i, name in enumerate(self.models.keys()) if i < len(probs)
            },
        }

    def predict_horizon(self, X_raw: np.ndarray, horizon: int, df: pd.DataFrame) -> Dict[str, Any]:
        if horizon not in self.horizon_models:
            return {}

        h_info = self.horizon_models[horizon]
        h_model = h_info["model"]
        h_cols = h_info["feature_cols"]
        h_scaler = self.horizon_scalers.get(horizon)

        if h_scaler is None:
            return {}

        try:
            latest = df[h_cols].iloc[-1:].ffill().fillna(0).replace([np.inf, -np.inf], 0)
            X_h = h_scaler.transform(latest.values.astype(np.float32))
            prob = float(h_model.predict_proba(X_h)[:, 1][0])
            direction = int(prob >= 0.5)
            return {
                "direction": direction,
                "probability": round(prob, 4),
                "accuracy": round(self.horizon_accuracies.get(horizon, 0), 4),
            }
        except Exception as e:
            logger.debug(f"Horizon {horizon}d prediction failed: {e}")
            return {}

    def _compute_conformal_intervals(self, avg_prob: float) -> Dict[str, Any]:
        if self.conformal_scores is None or not self.conformal_quantiles:
            return {}

        result = {}
        for level, q_val in self.conformal_quantiles.items():
            lower = max(0.0, avg_prob - q_val)
            upper = min(1.0, avg_prob + q_val)
            result[f"ci_{level}"] = {
                "lower": round(lower, 4),
                "upper": round(upper, 4),
                "width": round(upper - lower, 4),
            }
        return result

    def _get_shap(self, X_scaled: np.ndarray) -> list:
        try:
            import shap
            xgb_model = self.models.get("xgb")
            if xgb_model is None:
                return []
            explainer = shap.TreeExplainer(xgb_model)
            sv = explainer.shap_values(X_scaled)
            if isinstance(sv, list):
                sv = sv[1] if len(sv) > 1 else sv[0]
            if sv.ndim > 1:
                sv = sv[0]
            top_idx = np.argsort(np.abs(sv))[::-1][:8]
            return [
                {
                    "feature": self.feature_cols[i] if i < len(self.feature_cols) else f"f{i}",
                    "shap_value": round(float(sv[i]), 4),
                    "direction": "bullish" if sv[i] > 0 else "bearish",
                    "impact": round(abs(float(sv[i])), 4),
                }
                for i in top_idx if i < len(self.feature_cols)
            ]
        except Exception as e:
            logger.debug(f"SHAP failed: {e}")
            return []

    def check_staleness(self, current_accuracy: Optional[float] = None) -> Dict[str, Any]:
        age_hours = (time.time() - self.fitted_at) / 3600 if self.fitted_at > 0 else float('inf')
        is_stale = age_hours > STALENESS_MAX_AGE_HOURS

        accuracy_degraded = False
        if current_accuracy is not None and self.cv_accuracy > 0:
            accuracy_degraded = (self.cv_accuracy - current_accuracy) > STALENESS_ACCURACY_DROP

        needs_retrain = is_stale or accuracy_degraded

        self.staleness_info = {
            "age_hours": round(age_hours, 1),
            "is_stale": is_stale,
            "accuracy_degraded": accuracy_degraded,
            "needs_retrain": needs_retrain,
            "original_accuracy": round(self.cv_accuracy, 4),
            "current_accuracy": round(current_accuracy, 4) if current_accuracy is not None else None,
        }
        return self.staleness_info


class PredictionEngine:

    def __init__(self):
        self._models: Dict[str, ProductionEnsemble] = {}
        self._prediction_cache: Dict[str, Tuple[float, dict]] = {}
        self._inflight_lock = threading.Lock()
        self._inflight: Dict[str, threading.Event] = {}
        self._inflight_results: Dict[str, dict] = {}

    def _get_or_train(self, symbol: str, force_retrain: bool = False) -> Optional[ProductionEnsemble]:
        model_path = os.path.join(MODEL_DIR, f"{symbol}_ensemble_v2.joblib")

        if symbol in self._models and not force_retrain:
            ensemble = self._models[symbol]
            staleness = ensemble.check_staleness()
            if not staleness.get("needs_retrain", False):
                return ensemble
            logger.info(f"Model for {symbol} is stale (age={staleness['age_hours']}h), retraining...")

        if not force_retrain and os.path.exists(model_path):
            age = time.time() - os.path.getmtime(model_path)
            if age < 86400:
                try:
                    loaded = joblib.load(model_path)
                    if hasattr(loaded, 'models') and len(loaded.models) == 3:
                        for attr, default in [
                            ('conformal_scores', None),
                            ('conformal_quantiles', {}),
                            ('horizon_models', {}),
                            ('horizon_scalers', {}),
                            ('horizon_accuracies', {}),
                            ('feature_interactions', []),
                            ('staleness_info', {}),
                            ('fitted_at', time.time() - age),
                        ]:
                            if not hasattr(loaded, attr):
                                setattr(loaded, attr, default)
                        if not hasattr(loaded, 'wf_accuracy'):
                            loaded.wf_accuracy = getattr(loaded, 'cv_accuracy', 0.5)
                        self._models[symbol] = loaded
                        logger.info(f"Loaded cached 3-model ensemble for {symbol}")
                        return self._models[symbol]
                except Exception:
                    pass

        df = build_feature_matrix(symbol, days=DEFAULT_LOOKBACK)
        if df is None or len(df) < MIN_TRAIN_SIZE:
            return None

        ensemble = ProductionEnsemble(n_trials=OPTUNA_TRIALS)
        result = ensemble.fit(df)
        if "error" in result:
            return None

        self._models[symbol] = ensemble
        try:
            joblib.dump(ensemble, model_path)
        except Exception:
            pass

        return ensemble

    def predict(self, symbol: str) -> Dict[str, Any]:
        if symbol in self._prediction_cache:
            cached_time, cached_result = self._prediction_cache[symbol]
            if time.time() - cached_time < CACHE_TTL_PREDICTION:
                return cached_result

        with self._inflight_lock:
            if symbol in self._inflight:
                event = self._inflight[symbol]
            else:
                event = None

        if event is not None:
            completed = event.wait(timeout=120)
            if not completed:
                return {"error": f"Prediction for {symbol} timed out (in-flight dedup wait)"}
            if symbol in self._inflight_results:
                return self._inflight_results[symbol]
            if symbol in self._prediction_cache:
                cached_time, cached_result = self._prediction_cache[symbol]
                if time.time() - cached_time < CACHE_TTL_PREDICTION:
                    return cached_result
            return {"error": f"Prediction for {symbol} completed but result unavailable"}

        event = threading.Event()
        with self._inflight_lock:
            self._inflight[symbol] = event

        try:
            result = self._predict_impl(symbol)
            with self._inflight_lock:
                self._inflight_results[symbol] = result
            return result
        finally:
            event.set()
            time.sleep(0.01)
            with self._inflight_lock:
                self._inflight.pop(symbol, None)
                self._inflight_results.pop(symbol, None)

    def _predict_impl(self, symbol: str) -> Dict[str, Any]:
        df = build_feature_matrix(symbol, days=DEFAULT_LOOKBACK)
        if df is None or len(df) < 100:
            return {"error": f"Insufficient data for {symbol}"}

        ensemble = self._get_or_train(symbol)
        if ensemble is None or not ensemble.is_fitted:
            return {"error": f"Model training failed for {symbol}"}

        feature_cols = ensemble.feature_cols
        latest = df[feature_cols].iloc[-1:].ffill().fillna(0).replace([np.inf, -np.inf], 0)
        pred = ensemble.predict(latest.values[0])

        current_price = float(df["close"].iloc[-1])
        avg_daily_ret = float(df["close"].pct_change().tail(20).mean())
        vol_20d = float(df["close"].pct_change().tail(20).std())

        if pred["direction"] == 1:
            trend_pct = abs(avg_daily_ret * FORWARD_DAYS * 100) + vol_20d * 50
        else:
            trend_pct = -(abs(avg_daily_ret * FORWARD_DAYS * 100) + vol_20d * 50)
        trend_pct = round(np.clip(trend_pct, -20, 20), 2)

        vol_ann = float(vol_20d * np.sqrt(252))
        liquidity = round(float(df["volume"].tail(20).mean()) / 1e6, 1)
        edge = round(abs(trend_pct) * pred["confidence"] / 100 / max(vol_ann, 0.1), 1)
        sharpe_est = round(trend_pct / max(vol_ann * 100, 1) * np.sqrt(252 / FORWARD_DAYS), 2)

        if trend_pct > 3 and pred["confidence"] > 70:
            rec = "STRONG_BUY"
        elif trend_pct > 1.5 and pred["confidence"] > 55:
            rec = "BUY"
        elif trend_pct < -3 and pred["confidence"] > 70:
            rec = "STRONG_SELL"
        elif trend_pct < -1.5 and pred["confidence"] > 55:
            rec = "SELL"
        else:
            rec = "HOLD"

        direction_str = "bullish" if trend_pct > 0.5 else "bearish" if trend_pct < -0.5 else "neutral"
        risk_level = "LOW" if vol_ann < 0.2 else "MEDIUM" if vol_ann < 0.4 else "HIGH"

        change_1d = float(df["close"].pct_change().iloc[-1] * 100)
        change_5d = float((df["close"].iloc[-1] / df["close"].iloc[-5] - 1) * 100) if len(df) > 5 else 0

        ci_1d = vol_20d * np.sqrt(1) * current_price
        ci_5d = vol_20d * np.sqrt(5) * current_price
        ci_20d = vol_20d * np.sqrt(20) * current_price

        multi_horizon_predictions = {}
        for h in MULTI_HORIZONS:
            if h == 5:
                multi_horizon_predictions["5d"] = {
                    "direction": direction_str,
                    "probability": pred["probability"],
                    "accuracy": round(ensemble.cv_accuracy * 100, 1),
                    "expected_price": round(current_price * (1 + trend_pct / 100), 2),
                }
            else:
                h_pred = ensemble.predict_horizon(latest.values[0], h, df)
                if h_pred:
                    h_dir = "bullish" if h_pred["direction"] == 1 else "bearish"
                    h_trend = trend_pct * (h / 5.0)
                    multi_horizon_predictions[f"{h}d"] = {
                        "direction": h_dir,
                        "probability": h_pred["probability"],
                        "accuracy": round(h_pred.get("accuracy", 0) * 100, 1),
                        "expected_price": round(current_price * (1 + h_trend / 100), 2),
                    }

        staleness = ensemble.check_staleness()

        result = {
            "symbol": symbol,
            "price": round(current_price, 2),
            "change_1d": round(change_1d, 2),
            "change_5d": round(change_5d, 2),
            "predicted_trend_pct": trend_pct,
            "confidence": pred["confidence"],
            "probability": pred["probability"],
            "direction": direction_str,
            "recommendation": rec,
            "risk_level": risk_level,
            "edge_score": edge,
            "sharpe_estimate": sharpe_est,
            "liquidity_m": liquidity,
            "volatility_ann": round(vol_ann * 100, 1),
            "model_accuracy": round(ensemble.cv_accuracy * 100, 1),
            "wf_accuracy": round(getattr(ensemble, 'wf_accuracy', ensemble.cv_accuracy) * 100, 1),
            "regime": getattr(ensemble, 'regime', 'neutral'),
            "regime_weights": getattr(ensemble, 'regime_weights', {"momentum": 0.5, "value": 0.5}),
            "model_agreement": pred.get("model_agreement", 0),
            "ensemble_std": pred.get("ensemble_std", 0),
            "model_weights": {
                "xgboost": round(ensemble.model_weights[0], 3),
                "lightgbm": round(ensemble.model_weights[1], 3),
                "catboost": round(ensemble.model_weights[2], 3),
            },
            "feature_count": len(ensemble.feature_cols),
            "important_feature_count": len(getattr(ensemble, 'important_features', ensemble.feature_cols)),
            "fundamental_features_count": len(getattr(ensemble, 'fundamental_feature_importance', {})),
            "top_features": getattr(ensemble, 'feature_importance_ranked', [])[:10],
            "fundamental_feature_importance": getattr(ensemble, 'fundamental_feature_importance', {}),
            "feature_interactions": getattr(ensemble, 'feature_interactions', [])[:5],
            "shap_factors": pred.get("shap_factors", []),
            "model_probas": pred.get("model_probas", {}),
            "conformal_intervals": pred.get("conformal_intervals", {}),
            "multi_horizon": multi_horizon_predictions,
            "horizon_accuracies": {f"{k}d": round(v * 100, 1) for k, v in ensemble.horizon_accuracies.items()},
            "staleness": staleness,
            "forecast": {
                "1d": {
                    "expected_price": round(current_price * (1 + trend_pct / 100 / 5), 2),
                    "upper_95": round(current_price + ci_1d * 1.96, 2),
                    "lower_95": round(current_price - ci_1d * 1.96, 2),
                },
                "5d": {
                    "expected_price": round(current_price * (1 + trend_pct / 100), 2),
                    "upper_95": round(current_price + ci_5d * 1.96, 2),
                    "lower_95": round(current_price - ci_5d * 1.96, 2),
                },
                "20d": {
                    "expected_price": round(current_price * (1 + trend_pct / 100 * 4), 2),
                    "upper_95": round(current_price + ci_20d * 1.96, 2),
                    "lower_95": round(current_price - ci_20d * 1.96, 2),
                },
            },
            "technicals": self._extract_technicals(df),
        }

        self._prediction_cache[symbol] = (time.time(), result)
        return result

    def _extract_technicals(self, df: pd.DataFrame) -> dict:
        result = {}
        for ind in ["rsi", "adx", "macd_hist", "bb_pctb", "stoch_k", "cci", "atr_pct", "williams_r"]:
            if ind in df.columns:
                val = df[ind].iloc[-1]
                result[ind] = round(float(val), 2) if not np.isnan(val) else None
        return result


_engine_instance = None

def get_engine() -> PredictionEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = PredictionEngine()
    return _engine_instance


def warmup_models(tickers: list) -> None:
    def _warmup():
        engine = get_engine()
        for ticker in tickers:
            try:
                engine.predict(ticker)
                logger.info(f"Warmed up ML model for {ticker}")
            except Exception as e:
                logger.warning(f"Warmup failed for {ticker}: {e}")

    t = threading.Thread(target=_warmup, daemon=True)
    t.start()
