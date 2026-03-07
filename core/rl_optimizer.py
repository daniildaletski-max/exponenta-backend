"""
Reinforcement Learning portfolio optimizer using PPO (Proximal Policy Optimization).

Learns a trading/rebalancing policy that maximises risk-adjusted returns
in a simulated portfolio environment. Complements the MPT-based optimizer
by capturing non-linear dynamics and regime shifts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import structlog

log = structlog.get_logger()


@dataclass
class RLAction:
    ticker: str
    action: str      # "buy" | "sell" | "hold"
    magnitude: float  # fraction of available capital
    conviction: float


class RLPortfolioOptimizer:
    """
    PPO-based portfolio policy optimizer.

    Architecture:
      - State: [portfolio_weights, returns_window, volatility, sentiment_scores]
      - Action: continuous weight adjustments per asset
      - Reward: risk-adjusted return (Sharpe-like) with drawdown penalty
    """

    def __init__(self, model_path: str | None = None):
        self._model = None
        self._model_path = model_path

    def _load(self):
        if self._model is not None:
            return
        if self._model_path is None:
            log.info("rl_optimizer.no_model_path — using heuristic policy")
            return
        try:
            from stable_baselines3 import PPO
            self._model = PPO.load(self._model_path)
            log.info("rl_optimizer.loaded", path=self._model_path)
        except (ImportError, FileNotFoundError):
            log.warning("rl_optimizer.load_failed")

    def rank_actions(
        self,
        current_weights: dict[str, float],
        predicted_returns: dict[str, float],
        sentiment_scores: dict[str, float],
    ) -> list[RLAction]:
        """
        Rank rebalancing actions by expected utility.

        Falls back to a simple momentum + sentiment heuristic
        when the RL model is not available.
        """
        self._load()

        if self._model is not None:
            try:
                state_vec = []
                tickers = list(current_weights.keys())
                for t in tickers:
                    state_vec.extend([
                        current_weights.get(t, 0),
                        predicted_returns.get(t, 0),
                        sentiment_scores.get(t, 0),
                    ])
                obs = np.array(state_vec, dtype=np.float32)
                action, _ = self._model.predict(obs, deterministic=True)
                actions: list[RLAction] = []
                for i, ticker in enumerate(tickers):
                    act_val = float(action[i]) if i < len(action) else 0.0
                    if act_val > 0.1:
                        act_str = "buy"
                    elif act_val < -0.1:
                        act_str = "sell"
                    else:
                        act_str = "hold"
                    actions.append(RLAction(
                        ticker=ticker,
                        action=act_str,
                        magnitude=min(abs(act_val), 0.2),
                        conviction=min(abs(act_val) + 0.3, 1.0),
                    ))
                actions.sort(key=lambda a: a.conviction, reverse=True)
                return actions
            except Exception as e:
                log.warning("rl_optimizer.predict_failed", error=str(e))

        # Heuristic fallback: blend predicted return with sentiment
        actions: list[RLAction] = []
        for ticker in current_weights:
            pred_ret = predicted_returns.get(ticker, 0)
            sent = sentiment_scores.get(ticker, 0)
            score = 0.6 * pred_ret + 0.4 * sent

            if score > 0.3:
                action = "buy"
            elif score < -0.3:
                action = "sell"
            else:
                action = "hold"

            actions.append(RLAction(
                ticker=ticker,
                action=action,
                magnitude=min(abs(score), 0.2),
                conviction=min(abs(score) + 0.3, 1.0),
            ))

        actions.sort(key=lambda a: a.conviction, reverse=True)
        return actions
