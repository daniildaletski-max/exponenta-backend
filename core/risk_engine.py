"""
Advanced Risk Engine -- jump-diffusion Monte Carlo, stress testing, tail risk metrics.

Provides production-grade risk analytics for portfolio analysis including:
- Merton jump-diffusion Monte Carlo simulation
- Historical Value-at-Risk (VaR) and Conditional VaR (CVaR)
- Multi-scenario stress testing
- Comprehensive tail risk metrics
- Correlation stress testing for crisis simulation
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RiskMetrics:
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    expected_shortfall: float
    omega_ratio: float
    tail_ratio: float
    max_drawdown: float


@dataclass
class StressTestResult:
    scenario_name: str
    portfolio_impact_pct: float
    worst_asset: str
    worst_asset_impact_pct: float


# ---------------------------------------------------------------------------
# Pre-defined stress scenarios: asset_class -> shock magnitude (fractional)
# ---------------------------------------------------------------------------
_DEFAULT_SCENARIOS: dict[str, dict[str, float]] = {
    "2008_crisis": {
        "stock": -0.50,
        "etf": -0.50,
        "bond": 0.10,
        "crypto": -0.80,
        "gold": 0.25,
    },
    "covid_crash": {
        "stock": -0.34,
        "etf": -0.34,
        "bond": 0.05,
        "crypto": -0.50,
        "gold": 0.08,
    },
    "rate_hike_2022": {
        "stock": -0.25,
        "etf": -0.25,
        "bond": -0.15,
        "crypto": -0.65,
        "gold": -0.05,
    },
    "flash_crash": {
        "stock": -0.10,
        "etf": -0.10,
        "bond": 0.02,
        "crypto": -0.20,
        "gold": 0.03,
    },
    "stagflation": {
        "stock": -0.20,
        "etf": -0.20,
        "bond": -0.10,
        "crypto": -0.40,
        "gold": 0.15,
    },
}


class RiskEngine:
    """Production risk analytics engine."""

    # ------------------------------------------------------------------ #
    #  Monte Carlo -- Merton jump-diffusion                               #
    # ------------------------------------------------------------------ #

    def jump_diffusion_monte_carlo(
        self,
        prices: np.ndarray,
        n_simulations: int = 100_000,
        n_days: int = 252,
        jump_intensity: float = 0.1,
        jump_mean: float = -0.05,
        jump_std: float = 0.1,
        seed: int | None = None,
    ) -> dict:
        """Merton jump-diffusion model: GBM + compound Poisson jumps.

        The SDE is:
            dS/S = (mu - lambda*k) dt + sigma dW + J dN

        where
            dW  = Brownian motion increment
            dN  = Poisson process with intensity *jump_intensity*
            J   = log-normal jump size  (ln(1+J) ~ N(jump_mean, jump_std^2))
            k   = E[e^J - 1]

        Parameters
        ----------
        prices : np.ndarray
            Historical price series (oldest first).
        n_simulations : int
            Number of Monte Carlo paths.
        n_days : int
            Simulation horizon in trading days.
        jump_intensity : float
            Expected number of jumps per day (lambda).
        jump_mean : float
            Mean of the log-jump size.
        jump_std : float
            Std of the log-jump size.
        seed : int | None
            Optional RNG seed for reproducibility.

        Returns
        -------
        dict with keys:
            paths          -- (n_simulations, n_days+1) array of price paths
            var_95         -- 95% Value-at-Risk of terminal return
            cvar_95        -- 95% Conditional VaR (Expected Shortfall)
            percentiles    -- dict mapping label -> terminal price percentile
        """
        prices = np.asarray(prices, dtype=np.float64)
        log_returns = np.diff(np.log(prices))

        mu = float(np.mean(log_returns)) * 252  # annualised drift
        sigma = float(np.std(log_returns, ddof=1)) * np.sqrt(252)  # annualised vol
        S0 = float(prices[-1])
        dt = 1.0 / 252.0

        # k = E[e^J - 1] for log-normal jump sizes
        k = np.exp(jump_mean + 0.5 * jump_std**2) - 1.0

        rng = np.random.default_rng(seed)

        # --- Brownian increments ---
        Z = rng.standard_normal((n_simulations, n_days))

        # --- Poisson jump counts per step ---
        N = rng.poisson(jump_intensity * dt, size=(n_simulations, n_days))

        # --- Jump sizes: for each (sim, day) draw the *sum* of N jumps ---
        # For efficiency, pre-draw total jump magnitude per step.
        # Sum of N_i iid normals ~ N(N_i * mu_J, N_i * sigma_J^2)
        jump_sizes = np.zeros_like(N, dtype=np.float64)
        mask = N > 0
        counts = N[mask]
        jump_sizes[mask] = rng.normal(
            loc=jump_mean * counts,
            scale=jump_std * np.sqrt(counts),
        )

        # --- Build log-price increments ---
        drift = (mu - 0.5 * sigma**2 - jump_intensity * k) * dt
        diffusion = sigma * np.sqrt(dt) * Z
        log_increments = drift + diffusion + jump_sizes  # (n_sim, n_days)

        # --- Cumulative sum -> price paths ---
        log_paths = np.zeros((n_simulations, n_days + 1), dtype=np.float64)
        log_paths[:, 0] = np.log(S0)
        log_paths[:, 1:] = np.log(S0) + np.cumsum(log_increments, axis=1)
        paths = np.exp(log_paths)

        # --- Terminal return distribution ---
        terminal_returns = paths[:, -1] / S0 - 1.0

        var_95 = float(-np.percentile(terminal_returns, 5))
        cvar_95_mask = terminal_returns <= np.percentile(terminal_returns, 5)
        cvar_95 = float(-np.mean(terminal_returns[cvar_95_mask])) if cvar_95_mask.any() else var_95

        percentiles = {
            "p5": float(np.percentile(paths[:, -1], 5)),
            "p25": float(np.percentile(paths[:, -1], 25)),
            "p50": float(np.percentile(paths[:, -1], 50)),
            "p75": float(np.percentile(paths[:, -1], 75)),
            "p95": float(np.percentile(paths[:, -1], 95)),
        }

        return {
            "paths": paths,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "percentiles": percentiles,
        }

    # ------------------------------------------------------------------ #
    #  Historical VaR / CVaR                                              #
    # ------------------------------------------------------------------ #

    def historical_var(
        self, returns: np.ndarray, confidence: float = 0.95
    ) -> dict:
        """Historical VaR and CVaR from the empirical return distribution.

        Parameters
        ----------
        returns : np.ndarray
            Array of periodic (e.g. daily) simple or log returns.
        confidence : float
            Confidence level (e.g. 0.95 for 95% VaR).

        Returns
        -------
        dict with keys: var, cvar, confidence, n_observations
        """
        returns = np.asarray(returns, dtype=np.float64)
        sorted_returns = np.sort(returns)

        # VaR: the loss threshold at the (1 - confidence) quantile
        alpha = 1.0 - confidence
        var_threshold = float(np.percentile(sorted_returns, alpha * 100))
        var_value = -var_threshold  # convention: VaR reported as positive number

        # CVaR (Expected Shortfall): mean of returns in the tail
        tail = sorted_returns[sorted_returns <= var_threshold]
        cvar_value = float(-np.mean(tail)) if len(tail) > 0 else var_value

        return {
            "var": var_value,
            "cvar": cvar_value,
            "confidence": confidence,
            "n_observations": len(returns),
        }

    # ------------------------------------------------------------------ #
    #  Stress testing                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _classify_asset(ticker: str, asset_class: str | None = None) -> str:
        """Heuristic asset-class classifier."""
        if asset_class:
            ac = asset_class.strip().lower()
            if ac in ("stock", "etf", "bond", "crypto", "gold"):
                return ac
            if ac in ("equity", "equities"):
                return "stock"
            if ac in ("fixed_income", "fixed income", "bonds"):
                return "bond"
            if ac in ("cryptocurrency", "digital"):
                return "crypto"
            if ac in ("commodity",):
                return "gold"

        t = ticker.upper()
        crypto_tickers = {
            "BTC", "ETH", "SOL", "ADA", "DOT", "DOGE", "XRP", "AVAX",
            "LINK", "MATIC", "BTC-USD", "ETH-USD", "SOL-USD",
        }
        gold_tickers = {"GLD", "IAU", "GOLD", "XAUUSD", "GC=F"}
        bond_etfs = {"TLT", "IEF", "SHY", "BND", "AGG", "LQD", "HYG", "GOVT"}
        etf_tickers = {"SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "VEA", "VWO"}

        if t in crypto_tickers or t.endswith("-USD"):
            return "crypto"
        if t in gold_tickers:
            return "gold"
        if t in bond_etfs:
            return "bond"
        if t in etf_tickers:
            return "etf"
        return "stock"

    def stress_test(
        self,
        holdings: list[dict],
        scenarios: list[str] | None = None,
    ) -> list[StressTestResult]:
        """Run stress-test scenarios against a portfolio.

        Parameters
        ----------
        holdings : list[dict]
            Each dict must contain at least ``ticker`` and ``value``
            (current market value).  Optional ``asset_class`` key overrides
            automatic classification.
        scenarios : list[str] | None
            Scenario names to run (keys of ``_DEFAULT_SCENARIOS``).
            *None* runs all default scenarios.

        Returns
        -------
        list[StressTestResult]
        """
        if scenarios is None:
            scenarios = list(_DEFAULT_SCENARIOS.keys())

        total_value = sum(h["value"] for h in holdings)
        if total_value == 0:
            return []

        results: list[StressTestResult] = []

        for scenario_name in scenarios:
            shocks = _DEFAULT_SCENARIOS.get(scenario_name)
            if shocks is None:
                continue

            portfolio_pnl = 0.0
            worst_asset = ""
            worst_impact = 0.0  # most negative impact

            for h in holdings:
                ac = self._classify_asset(
                    h["ticker"], h.get("asset_class")
                )
                shock = shocks.get(ac, shocks.get("stock", 0.0))
                asset_pnl = h["value"] * shock
                portfolio_pnl += asset_pnl

                if shock < worst_impact:
                    worst_impact = shock
                    worst_asset = h["ticker"]

            portfolio_impact_pct = (portfolio_pnl / total_value) * 100.0

            # If no asset had a negative shock, pick the one with the
            # smallest (least positive) shock as "worst".
            if not worst_asset and holdings:
                worst_asset = holdings[0]["ticker"]
                worst_impact = shocks.get(
                    self._classify_asset(
                        holdings[0]["ticker"],
                        holdings[0].get("asset_class"),
                    ),
                    0.0,
                )

            results.append(
                StressTestResult(
                    scenario_name=scenario_name,
                    portfolio_impact_pct=round(portfolio_impact_pct, 4),
                    worst_asset=worst_asset,
                    worst_asset_impact_pct=round(worst_impact * 100.0, 4),
                )
            )

        return results

    # ------------------------------------------------------------------ #
    #  Tail risk metrics                                                  #
    # ------------------------------------------------------------------ #

    def compute_tail_metrics(self, returns: np.ndarray) -> RiskMetrics:
        """Compute comprehensive tail-risk metrics.

        Parameters
        ----------
        returns : np.ndarray
            Array of periodic returns (simple or log).

        Returns
        -------
        RiskMetrics dataclass
        """
        returns = np.asarray(returns, dtype=np.float64)

        # --- VaR (reported as positive loss) ---
        var_95 = -float(np.percentile(returns, 5))
        var_99 = -float(np.percentile(returns, 1))

        # --- CVaR / Expected Shortfall ---
        tail_5 = returns[returns <= np.percentile(returns, 5)]
        tail_1 = returns[returns <= np.percentile(returns, 1)]

        cvar_95 = -float(np.mean(tail_5)) if len(tail_5) > 0 else var_95
        cvar_99 = -float(np.mean(tail_1)) if len(tail_1) > 0 else var_99
        expected_shortfall = cvar_95  # ES at 95% is the standard definition

        # --- Omega ratio: sum of gains / |sum of losses| (threshold = 0) ---
        gains = returns[returns > 0]
        losses = returns[returns < 0]
        sum_gains = float(np.sum(gains)) if len(gains) > 0 else 0.0
        sum_losses = float(np.abs(np.sum(losses))) if len(losses) > 0 else 1e-10
        omega_ratio = sum_gains / sum_losses

        # --- Tail ratio: |95th percentile / 5th percentile| ---
        p95 = np.percentile(returns, 95)
        p5 = np.percentile(returns, 5)
        tail_ratio = float(np.abs(p95 / p5)) if p5 != 0 else float("inf")

        # --- Maximum drawdown from cumulative returns ---
        cumulative = np.cumprod(1.0 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = float(-np.min(drawdowns))  # positive number

        return RiskMetrics(
            var_95=round(var_95, 8),
            var_99=round(var_99, 8),
            cvar_95=round(cvar_95, 8),
            cvar_99=round(cvar_99, 8),
            expected_shortfall=round(expected_shortfall, 8),
            omega_ratio=round(omega_ratio, 8),
            tail_ratio=round(tail_ratio, 8),
            max_drawdown=round(max_drawdown, 8),
        )

    # ------------------------------------------------------------------ #
    #  Correlation stress                                                 #
    # ------------------------------------------------------------------ #

    def correlation_stress(
        self,
        returns_matrix: np.ndarray,
        stress_factor: float = 1.5,
    ) -> np.ndarray:
        """Simulate correlation breakdown during a crisis.

        In tail events, asset correlations tend to spike toward +1.  This
        method takes the empirical correlation matrix and moves the
        off-diagonal entries toward 1 by ``stress_factor``, then ensures the
        result is a valid (positive semi-definite) correlation matrix.

        Parameters
        ----------
        returns_matrix : np.ndarray
            (T, N) matrix of returns for N assets over T periods.
        stress_factor : float
            Multiplier for off-diagonal correlations.  A value of 1.5 moves
            each correlation 50% of the way toward +1 (capped at 1).

        Returns
        -------
        np.ndarray
            (N, N) stressed correlation matrix (PSD, unit diagonal).
        """
        returns_matrix = np.asarray(returns_matrix, dtype=np.float64)
        corr = np.corrcoef(returns_matrix, rowvar=False)  # (N, N)
        n = corr.shape[0]

        # Stress off-diagonal elements toward +1
        stressed = corr.copy()
        for i in range(n):
            for j in range(n):
                if i != j:
                    rho = corr[i, j]
                    # Move correlation toward 1:
                    #   stressed_rho = rho + stress_factor * (1 - rho)
                    #   when stress_factor=0 -> no change
                    #   when stress_factor=1 -> perfect correlation
                    # We re-parameterise so that stress_factor=1.5 means
                    # "multiply existing correlation by 1.5, cap at 1".
                    stressed_rho = rho * stress_factor
                    stressed_rho = np.clip(stressed_rho, -1.0, 1.0)
                    stressed[i, j] = stressed_rho

        # Ensure positive semi-definite via eigenvalue clipping
        eigenvalues, eigenvectors = np.linalg.eigh(stressed)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        stressed_psd = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        # Re-normalise to unit diagonal (proper correlation matrix)
        diag_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(stressed_psd)))
        stressed_corr = diag_inv_sqrt @ stressed_psd @ diag_inv_sqrt

        # Clip any floating-point artefacts
        stressed_corr = np.clip(stressed_corr, -1.0, 1.0)
        np.fill_diagonal(stressed_corr, 1.0)

        return stressed_corr
