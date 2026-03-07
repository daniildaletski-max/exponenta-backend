"""
RunPod serverless GPU worker for Exponenta.

Provides persistent GPU endpoint or serverless inference.

Setup:
  1. Create account at runpod.io
  2. Build Docker image with this handler
  3. Create Serverless Endpoint
  4. Set RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID in .env
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import structlog

log = structlog.get_logger()

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID", "")
RUNPOD_BASE_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}"


async def call_runpod(task: str, payload: dict, timeout: float = 120.0) -> dict:
    """
    Call RunPod serverless endpoint.

    Args:
        task: Task identifier (e.g., "monte_carlo", "train_lstm", "train_tft").
        payload: Input data for the GPU worker.
        timeout: Request timeout in seconds.

    Returns:
        GPU computation results.
    """
    if not RUNPOD_API_KEY or not RUNPOD_ENDPOINT_ID:
        raise RuntimeError("RunPod not configured: set RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID")

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{RUNPOD_BASE_URL}/runsync",
            headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
            json={"input": {"task": task, **payload}},
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()

        if data.get("status") == "FAILED":
            raise RuntimeError(f"RunPod job failed: {data.get('error', 'unknown')}")

        return data.get("output", {})


async def call_runpod_async(task: str, payload: dict) -> str:
    """
    Submit async job to RunPod. Returns job ID for polling.
    """
    if not RUNPOD_API_KEY or not RUNPOD_ENDPOINT_ID:
        raise RuntimeError("RunPod not configured")

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{RUNPOD_BASE_URL}/run",
            headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
            json={"input": {"task": task, **payload}},
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()["id"]


async def poll_runpod_job(job_id: str, timeout: float = 300.0) -> dict:
    """Poll for RunPod job completion."""
    import asyncio

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout

    async with httpx.AsyncClient() as client:
        while loop.time() < deadline:
            response = await client.get(
                f"{RUNPOD_BASE_URL}/status/{job_id}",
                headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
                timeout=10.0,
            )
            data = response.json()

            if data["status"] == "COMPLETED":
                return data.get("output", {})
            if data["status"] == "FAILED":
                raise RuntimeError(f"RunPod job {job_id} failed: {data.get('error')}")

            await asyncio.sleep(2)

    raise TimeoutError(f"RunPod job {job_id} timed out after {timeout}s")


# ── RunPod Handler (for building the Docker image) ──────────────────

def _handler(event: dict[str, Any]) -> dict:
    """
    RunPod serverless handler entry point.

    Build as Docker image and deploy to RunPod:
        docker build -f Dockerfile.runpod -t exponenta-gpu .
    """
    import numpy as np

    input_data = event.get("input", {})
    task = input_data.get("task", "monte_carlo")

    if task == "monte_carlo":
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if "prices" not in input_data or not input_data["prices"]:
            return {"error": "Missing 'prices' in input"}

        prices = np.array(input_data["prices"], dtype=np.float32)
        n_sims = min(int(input_data.get("n_simulations", 100_000)), 2_000_000)
        n_days = min(int(input_data.get("n_days", 252)), 504)

        if len(prices) < 2:
            return {"error": "Need at least 2 price points"}

        returns = np.diff(np.log(prices))
        mu, sigma = float(np.mean(returns)), float(np.std(returns))

        z = torch.randn(n_sims, n_days, device=device)
        # Apply Ito correction for GBM
        daily_returns = (mu - 0.5 * sigma ** 2) + sigma * z
        paths = torch.exp(torch.cumsum(daily_returns, dim=1)) * float(prices[-1])
        final_prices = paths[:, -1].cpu().numpy()

        return {
            "percentiles": {
                str(p): round(float(np.percentile(final_prices, p)), 2)
                for p in [5, 25, 50, 75, 95]
            },
            "mean": round(float(final_prices.mean()), 2),
            "device": str(device),
        }

    elif task == "train_lstm":
        return {"status": "LSTM training not yet implemented on RunPod"}

    elif task == "train_tft":
        return {"status": "TFT training not yet implemented on RunPod"}

    return {"error": f"Unknown task: {task}"}


if __name__ == "__main__":
    try:
        import runpod
        runpod.serverless.start({"handler": _handler})
    except ImportError:
        print("RunPod SDK not installed. Run: pip install runpod")
