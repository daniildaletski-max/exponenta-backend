import asyncio
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

_batch_cache: Dict[str, Dict[str, Any]] = {}
_batch_cache_ts: Dict[str, float] = {}
BATCH_CACHE_TTL = 1800

XAI_BATCH_URL = "https://api.x.ai/v1/batches"


def _xai_headers() -> dict:
    key = os.environ.get("XAI_API_KEY", "")
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def get_cached_result(ticker: str) -> Optional[Dict[str, Any]]:
    key = ticker.upper()
    if key in _batch_cache:
        if time.time() - _batch_cache_ts.get(key, 0) < BATCH_CACHE_TTL:
            return _batch_cache[key]
        else:
            del _batch_cache[key]
            del _batch_cache_ts[key]
    return None


def set_cached_result(ticker: str, result: Dict[str, Any]):
    key = ticker.upper()
    _batch_cache[key] = result
    _batch_cache_ts[key] = time.time()


def get_all_cached() -> Dict[str, Any]:
    now = time.time()
    valid = {}
    for key, val in list(_batch_cache.items()):
        if now - _batch_cache_ts.get(key, 0) < BATCH_CACHE_TTL:
            valid[key] = val
    return valid


def _build_prompt(ticker: str, analysis_type: str) -> str:
    ticker = ticker.upper()
    if analysis_type == "sentiment":
        return (
            f"Analyze current market sentiment for {ticker} stock. "
            f"Consider recent news, social media, technical indicators, and market trends. "
            f"Return valid JSON only:\n"
            f'{{"ticker": "{ticker}", '
            f'"sentiment_score": <0-100>, '
            f'"sentiment": "bullish" | "neutral" | "bearish", '
            f'"key_catalysts": ["<catalyst1>", "<catalyst2>", "<catalyst3>"], '
            f'"risk_factors": ["<risk1>", "<risk2>"], '
            f'"short_term_outlook": "<2-3 sentence outlook>", '
            f'"confidence": <50-99>}}'
        )
    elif analysis_type == "fundamental":
        return (
            f"Perform fundamental analysis of {ticker} stock. "
            f"Consider valuation, growth, profitability, and competitive position. "
            f"Return valid JSON only:\n"
            f'{{"ticker": "{ticker}", '
            f'"fundamental_score": <0-100>, '
            f'"catalysts": ["<catalyst1>", "<catalyst2>", "<catalyst3>"], '
            f'"risks": ["<risk1>", "<risk2>"], '
            f'"sector_outlook": "positive" | "neutral" | "negative", '
            f'"fair_value_assessment": "undervalued" | "fairly_valued" | "overvalued", '
            f'"key_insight": "<2-3 sentence insight>"}}'
        )
    else:
        return (
            f"Provide a comprehensive investment analysis for {ticker}. "
            f"Return valid JSON only:\n"
            f'{{"ticker": "{ticker}", '
            f'"action": "STRONG_BUY" | "BUY" | "HOLD" | "SELL" | "STRONG_SELL", '
            f'"conviction": <1-10>, '
            f'"reasoning": "<2-3 sentence reasoning>", '
            f'"time_horizon": "1-5 days" | "1-4 weeks" | "1-3 months", '
            f'"key_risk": "<main risk>"}}'
        )


async def create_batch(name: str) -> Dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                XAI_BATCH_URL,
                headers=_xai_headers(),
                json={"name": name},
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "batch_id": data["batch_id"],
                "name": data.get("name", name),
                "status": "created",
                "state": data.get("state", {}),
            }
    except Exception as e:
        logger.error(f"Batch create failed: {e}")
        return {"error": str(e), "name": name, "status": "failed"}


async def submit_batch_analysis(
    batch_id: str,
    tickers: List[str],
    analysis_type: str = "sentiment",
) -> Dict[str, Any]:
    try:
        batch_requests = []
        for ticker in tickers:
            ticker = ticker.upper()
            prompt = _build_prompt(ticker, analysis_type)
            batch_requests.append({
                "batch_request_id": f"{analysis_type}_{ticker}",
                "batch_request": {
                    "chat_get_completion": {
                        "model": "grok-4",
                        "messages": [
                            {"role": "system", "content": "You are a senior financial analyst. Respond with valid JSON only."},
                            {"role": "user", "content": prompt},
                        ],
                        "max_tokens": 1000,
                        "temperature": 0.1,
                    }
                }
            })

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{XAI_BATCH_URL}/{batch_id}/requests",
                headers=_xai_headers(),
                json={"batch_requests": batch_requests},
            )
            resp.raise_for_status()

        return {
            "batch_id": batch_id,
            "tickers_submitted": tickers,
            "analysis_type": analysis_type,
            "request_count": len(batch_requests),
            "status": "submitted",
        }
    except Exception as e:
        logger.error(f"Batch submit failed: {e}")
        return {"error": str(e), "batch_id": batch_id, "status": "failed"}


async def check_batch_status(batch_id: str) -> Dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{XAI_BATCH_URL}/{batch_id}",
                headers=_xai_headers(),
            )
            resp.raise_for_status()
            data = resp.json()
            state = data.get("state", {})
            total = state.get("num_requests", 0)
            done = state.get("num_success", 0) + state.get("num_error", 0)
            is_complete = total > 0 and done >= total
            return {
                "batch_id": data["batch_id"],
                "name": data.get("name", ""),
                "state": state,
                "is_complete": is_complete,
                "progress": f"{done}/{total}",
                "create_time": data.get("create_time", ""),
                "expire_time": data.get("expire_time", ""),
            }
    except Exception as e:
        logger.error(f"Batch status check failed: {e}")
        return {"error": str(e), "batch_id": batch_id}


async def fetch_batch_results(batch_id: str) -> Dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{XAI_BATCH_URL}/{batch_id}/results",
                headers=_xai_headers(),
            )
            resp.raise_for_status()
            data = resp.json()

        results = {}
        errors = []
        batch_results = data.get("results", data.get("batch_results", []))

        for item in batch_results:
            req_id = item.get("batch_request_id", "")
            batch_result = item.get("batch_result", {})
            error = batch_result.get("error") or item.get("error")
            if error:
                errors.append({"request_id": req_id, "error": str(error)})
                continue
            try:
                response_data = batch_result.get("response", {})
                chat_resp = response_data.get("chat_get_completion", response_data)
                choices = chat_resp.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "")
                else:
                    content = str(chat_resp)

                content = content.strip()
                if content.startswith("```"):
                    parts = content.split("```")
                    if len(parts) >= 2:
                        block = parts[1]
                        if block.startswith("json"):
                            block = block[4:]
                        content = block.strip()
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    match = re.search(r'\{[\s\S]*\}', content)
                    if match:
                        parsed = json.loads(match.group())
                    else:
                        errors.append({"request_id": req_id, "error": "JSON parse failed"})
                        continue

                ticker = parsed.get("ticker", req_id.split("_")[-1] if "_" in req_id else req_id)
                ticker = ticker.upper()
                parsed["_batch_request_id"] = req_id
                parsed["_source"] = "batch_grok4"
                parsed["_cached_at"] = time.time()

                set_cached_result(ticker, parsed)
                results[ticker] = parsed
            except Exception as e:
                errors.append({"request_id": req_id, "error": str(e)})

        return {
            "batch_id": batch_id,
            "results_count": len(results),
            "errors_count": len(errors),
            "results": results,
            "errors": errors if errors else None,
            "cached": True,
        }
    except Exception as e:
        logger.error(f"Batch results fetch failed: {e}")
        return {"error": str(e), "batch_id": batch_id, "results_count": 0}


async def run_full_batch(
    tickers: List[str],
    batch_name: str = "exponenta_analysis",
    analysis_type: str = "sentiment",
    poll_interval: int = 10,
    max_wait: int = 300,
) -> Dict[str, Any]:
    batch_info = await create_batch(batch_name)
    if "error" in batch_info:
        return batch_info

    batch_id = batch_info["batch_id"]

    submit_result = await submit_batch_analysis(batch_id, tickers, analysis_type)
    if "error" in submit_result:
        return submit_result

    start_time = time.time()
    while time.time() - start_time < max_wait:
        await asyncio.sleep(poll_interval)
        status = await check_batch_status(batch_id)
        if status.get("is_complete"):
            break

    results = await fetch_batch_results(batch_id)
    results["batch_name"] = batch_name
    return results
