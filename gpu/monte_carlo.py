"""
GPU-accelerated Monte Carlo simulation using PyTorch.

Runs 100K+ price path simulations in parallel on GPU (CUDA / MPS / CPU fallback).
"""

from __future__ import annotations

import numpy as np
import torch
import structlog

from gpu.device import get_device

log = structlog.get_logger()


def gpu_monte_carlo(
    prices: np.ndarray,
    n_simulations: int = 100_000,
    n_days: int = 252,
    seed: int | None = None,
) -> dict:
    """
    Run massive Monte Carlo simulation for price paths.

    Args:
        prices: Historical daily close prices.
        n_simulations: Number of simulation paths (100K default, up to 1M on GPU).
        n_days: Trading days to simulate forward.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with percentiles, paths summary, VaR/CVaR.
    """
    device = get_device()
    log.info("monte_carlo.start", device=str(device), n_sims=n_simulations, n_days=n_days)

    if seed is not None:
        torch.manual_seed(seed)

    returns = np.diff(np.log(prices))
    mu = float(np.mean(returns))
    sigma = float(np.std(returns))
    initial_price = float(prices[-1])

    z = torch.randn(n_simulations, n_days, device=device, dtype=torch.float32)
    # Apply Ito correction: drift = mu - 0.5*sigma^2 for GBM
    daily_returns = (mu - 0.5 * sigma ** 2) + sigma * z

    paths = torch.exp(torch.cumsum(daily_returns, dim=1)) * initial_price

    final_prices = paths[:, -1].cpu().numpy()

    sorted_finals = np.sort(final_prices)
    var_95 = initial_price - float(np.percentile(sorted_finals, 5))
    cvar_95 = initial_price - float(np.mean(sorted_finals[sorted_finals <= np.percentile(sorted_finals, 5)]))

    sample_idx = np.linspace(0, n_simulations - 1, min(100, n_simulations), dtype=int)
    sample_paths = paths[sample_idx].cpu().numpy()

    result = {
        "percentiles": {
            "1st": round(float(np.percentile(final_prices, 1)), 2),
            "5th": round(float(np.percentile(final_prices, 5)), 2),
            "10th": round(float(np.percentile(final_prices, 10)), 2),
            "25th": round(float(np.percentile(final_prices, 25)), 2),
            "50th": round(float(np.percentile(final_prices, 50)), 2),
            "75th": round(float(np.percentile(final_prices, 75)), 2),
            "90th": round(float(np.percentile(final_prices, 90)), 2),
            "95th": round(float(np.percentile(final_prices, 95)), 2),
            "99th": round(float(np.percentile(final_prices, 99)), 2),
        },
        "mean": round(float(final_prices.mean()), 2),
        "std": round(float(final_prices.std()), 2),
        "initial_price": initial_price,
        "var_95": round(var_95, 2),
        "cvar_95": round(cvar_95, 2),
        "n_simulations": n_simulations,
        "n_days": n_days,
        "device": str(device),
        "sample_paths": sample_paths[:10].tolist(),
    }

    log.info("monte_carlo.done", device=str(device), mean=result["mean"])
    return result
