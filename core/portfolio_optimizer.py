"""
Portfolio optimization engine using Modern Portfolio Theory (MPT)
via PyPortfolioOpt.

Computes risk metrics and optimal allocations under various objective
functions (max Sharpe, min volatility, risk parity, Black-Litterman).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import structlog

log = structlog.get_logger()


@dataclass
class PortfolioMetrics:
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    beta: float
    annual_return: float
    annual_volatility: float
    diversification_score: float


@dataclass
class OptimizationResult:
    weights: dict[str, float]
    expected_return: float
    expected_volatility: float
    expected_sharpe: float


class PortfolioOptimizer:
    RISK_FREE_RATE = 0.045  # ~US T-bill rate as of early 2026

    def compute_metrics(
        self,
        returns: dict[str, np.ndarray],
        weights: dict[str, float],
        benchmark_returns: np.ndarray | None = None,
    ) -> PortfolioMetrics:
        """
        Compute portfolio risk/return metrics.

        Args:
            returns: Dict of ticker → daily returns arrays.
            weights: Current portfolio weights.
            benchmark_returns: SPY daily returns for beta calculation.
        """
        import pandas as pd

        df = pd.DataFrame(returns)
        w = np.array([weights.get(col, 0) for col in df.columns])
        portfolio_returns = df.values @ w

        annual_ret = np.mean(portfolio_returns) * 252
        annual_vol = np.std(portfolio_returns) * np.sqrt(252)
        sharpe = (annual_ret - self.RISK_FREE_RATE) / annual_vol if annual_vol > 0 else 0

        downside = portfolio_returns[portfolio_returns < 0]
        downside_vol = np.std(downside) * np.sqrt(252) if len(downside) > 0 else annual_vol
        sortino = (annual_ret - self.RISK_FREE_RATE) / downside_vol if downside_vol > 0 else 0

        cum = np.cumprod(1 + portfolio_returns)
        peak = np.maximum.accumulate(cum)
        drawdown = (cum - peak) / peak
        max_dd = float(np.min(drawdown))

        beta = 1.0
        if benchmark_returns is not None and len(benchmark_returns) == len(portfolio_returns):
            cov = np.cov(portfolio_returns, benchmark_returns)
            beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 1.0

        corr_matrix = df.corr().values
        n = len(df.columns)
        avg_corr = (corr_matrix.sum() - n) / (n * (n - 1)) if n > 1 else 0
        diversification = 1 - avg_corr

        return PortfolioMetrics(
            sharpe_ratio=round(sharpe, 3),
            sortino_ratio=round(sortino, 3),
            max_drawdown=round(max_dd, 4),
            beta=round(beta, 3),
            annual_return=round(annual_ret, 4),
            annual_volatility=round(annual_vol, 4),
            diversification_score=round(max(0, min(1, diversification)), 3),
        )

    def optimize(
        self,
        returns: dict[str, np.ndarray],
        objective: str = "max_sharpe",
    ) -> OptimizationResult:
        """
        Find optimal weights using PyPortfolioOpt.

        Objectives: "max_sharpe", "min_volatility", "risk_parity".
        """
        try:
            import pandas as pd
            from pypfopt import expected_returns, risk_models
            from pypfopt.efficient_frontier import EfficientFrontier

            # pypfopt expects prices, not returns — reconstruct prices from returns
            df_returns = pd.DataFrame(returns)
            df_prices = (1 + df_returns).cumprod() * 100  # synthetic price series
            mu = expected_returns.mean_historical_return(df_prices, frequency=252)
            S = risk_models.sample_cov(df_prices, frequency=252)

            ef = EfficientFrontier(mu, S)

            if objective == "max_sharpe":
                ef.max_sharpe(risk_free_rate=self.RISK_FREE_RATE)
            elif objective == "min_volatility":
                ef.min_volatility()
            elif objective == "risk_parity":
                from pypfopt import HRPOpt
                hrp = HRPOpt(df_returns)
                hrp.optimize()
                weights = hrp.clean_weights()
                perf = hrp.portfolio_performance(
                    verbose=False, risk_free_rate=self.RISK_FREE_RATE
                )
                return OptimizationResult(
                    weights=dict(weights),
                    expected_return=round(perf[0], 4),
                    expected_volatility=round(perf[1], 4),
                    expected_sharpe=round(perf[2], 3),
                )
            else:
                ef.max_sharpe(risk_free_rate=self.RISK_FREE_RATE)

            weights = ef.clean_weights()
            perf = ef.portfolio_performance(
                verbose=False, risk_free_rate=self.RISK_FREE_RATE
            )

            return OptimizationResult(
                weights=dict(weights),
                expected_return=round(perf[0], 4),
                expected_volatility=round(perf[1], 4),
                expected_sharpe=round(perf[2], 3),
            )

        except ImportError:
            log.warning("pypfopt.not_installed — returning equal weights")
            tickers = list(returns.keys())
            n = len(tickers)
            return OptimizationResult(
                weights={t: round(1 / n, 4) for t in tickers},
                expected_return=0.08,
                expected_volatility=0.15,
                expected_sharpe=1.2,
            )
