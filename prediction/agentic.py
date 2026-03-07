import asyncio
import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List

import httpx

logger = logging.getLogger(__name__)

SECTOR_ANALYSIS_TEMPLATES = {
    "Technology": {
        "key_metrics": ["revenue_growth", "R&D_spend", "TAM_expansion", "cloud_ARR", "AI_exposure"],
        "focus": "Evaluate TAM expansion, competitive moat (patents, network effects), R&D efficiency, cloud/AI revenue mix, and developer ecosystem strength.",
        "valuation_method": "DCF with revenue multiple comparison (EV/Revenue, EV/GP). High-growth tech often trades on forward revenue; focus on Rule of 40 (growth + margin).",
        "risk_focus": "Regulatory risk (antitrust, AI regulation), concentration risk in key products, talent retention, and technology obsolescence cycle.",
    },
    "Financial Services": {
        "key_metrics": ["NIM", "loan_growth", "credit_quality", "CET1_ratio", "efficiency_ratio"],
        "focus": "Net interest margin trajectory, credit quality trends (NPL ratio, charge-offs), capital adequacy, and fee income diversification.",
        "valuation_method": "P/B ratio relative to ROE, dividend yield sustainability. Banks trade on book value; assess tangible book value per share growth.",
        "risk_focus": "Interest rate sensitivity, credit cycle positioning, regulatory capital requirements, and CRE exposure.",
    },
    "Healthcare": {
        "key_metrics": ["pipeline_value", "patent_cliff", "FDA_approvals", "pricing_power"],
        "focus": "Pipeline optionality (Phase 1/2/3 catalysts), patent expiration timeline, M&A probability, and pricing/reimbursement dynamics.",
        "valuation_method": "Sum-of-parts with risk-adjusted pipeline NPV. Mature pharma: P/E with dividend yield. Biotech: pipeline valuation with probability-weighted outcomes.",
        "risk_focus": "Binary FDA outcomes, patent cliff timing, political pricing pressure, and clinical trial failure risk.",
    },
    "Consumer Cyclical": {
        "key_metrics": ["same_store_sales", "margin_expansion", "inventory_turns", "digital_mix"],
        "focus": "Consumer spending trends, brand strength, digital transformation progress, and supply chain efficiency.",
        "valuation_method": "P/E relative to growth, EV/EBITDA. Focus on margin trajectory and same-store sales momentum.",
        "risk_focus": "Consumer sentiment shifts, inventory buildup, competitive disruption from DTC brands, and input cost inflation.",
    },
    "Energy": {
        "key_metrics": ["production_growth", "breakeven_price", "reserve_life", "FCF_yield"],
        "focus": "Production growth trajectory, breakeven economics, reserve replacement ratio, and capital discipline.",
        "valuation_method": "EV/EBITDA, FCF yield, and NAV based on proved reserves. Compare to commodity price curve.",
        "risk_focus": "Commodity price volatility, ESG/transition risk, regulatory changes, and geopolitical supply disruption.",
    },
    "Communication Services": {
        "key_metrics": ["subscriber_growth", "ARPU", "content_spend", "ad_revenue"],
        "focus": "Subscriber/user growth and engagement metrics, ARPU trends, content investment ROI, and advertising market share.",
        "valuation_method": "EV/EBITDA, P/E with DCF for mature names. Streaming: EV/subscriber with path to profitability analysis.",
        "risk_focus": "Content cost inflation, subscriber churn, regulatory scrutiny, and advertising cycle sensitivity.",
    },
}

DEFAULT_SECTOR_TEMPLATE = {
    "key_metrics": ["revenue_growth", "margins", "cash_flow", "market_share"],
    "focus": "Evaluate competitive positioning, margin trajectory, cash flow generation, and market share dynamics.",
    "valuation_method": "P/E and EV/EBITDA relative to sector peers and historical ranges.",
    "risk_focus": "Competitive threats, macro sensitivity, regulatory changes, and execution risk.",
}

STRUCTURED_REASONING_PROMPT = """You are a senior financial analyst at a top-tier investment bank. You MUST structure your reasoning through these sections before giving your final JSON answer:

1. **Macro Environment**: Consider interest rates, current market regime (bull/bear/sideways), sector rotation dynamics, and macroeconomic indicators. How does the current cycle position affect this asset?
2. **Company Fundamentals**: Assess revenue growth trajectory, earnings quality (recurring vs one-time), valuation multiples vs peers and historical range, balance sheet strength, and management execution track record.
3. **Technical Analysis**: Evaluate price trend structure, key support/resistance levels, momentum indicators (RSI, MACD, ADX), volume confirmation, and pattern recognition signals.
4. **Catalyst Assessment**: Identify upcoming binary events — earnings dates, product launches, regulatory decisions, macro events, and sector-specific catalysts with expected timing.
5. **Risk Factors**: What could go wrong — competitive threats, valuation compression risk, macro headwinds, execution risk, and tail risk scenarios. Quantify downside where possible.
6. **Final Verdict**: Provide your conviction level (1-10) and clear directional call with specific price targets.

SCORING RUBRIC (use this to calibrate your fundamental_score):
- 90-100: Exceptional — clear catalyst, strong momentum, deep value, minimal risk
- 70-89: Strong — favorable setup with identifiable edge, manageable risks
- 50-69: Neutral — balanced risk/reward, no clear edge, wait for confirmation
- 30-49: Weak — deteriorating fundamentals or technicals, elevated risk
- 0-29: Avoid — significant fundamental or technical breakdown, high probability of loss

After your reasoning, respond with valid JSON only."""

SYNTHESIS_REASONING_PROMPT = """You are a portfolio manager synthesizing multiple analysis signals. Structure your thinking:

1. **Signal Alignment**: How well do quantitative (ML models) and qualitative (fundamental analysis) signals agree? Where do they diverge and why?
2. **Macro Context**: How does the current macro backdrop (rates, inflation, growth cycle) affect this position's risk/reward?
3. **Conviction Calibration**: What is your true conviction given model uncertainty, data quality, and market regime? Be brutally honest about what you don't know.
4. **Position Sizing Logic**: Given the risk metrics, what position size is appropriate? Use Kelly Criterion as a ceiling, not a target.
5. **Scenario Analysis**: What does the bull case look like (+2σ)? Bear case (-2σ)? Base case? Assign rough probabilities.
6. **Final Verdict**: Synthesize all signals into a single actionable recommendation with conviction 1-10. If signals conflict, default to caution.

Be calibrated and honest about uncertainty. Contrarian positions require higher conviction thresholds. Respond with valid JSON only."""

CONTRARIAN_PROMPT = """You are a contrarian analyst whose job is to stress-test the consensus view. Your role is to find the strongest argument AGAINST the prevailing analysis.

Given the consensus analysis below, construct the strongest possible counter-argument. Consider:
1. What assumptions does the consensus rely on that could be wrong?
2. What historical analogies suggest a different outcome?
3. What risks are being underweighted or ignored?
4. What would make this trade fail catastrophically?
5. Is there a timing risk — right thesis but wrong entry?

Be specific, data-driven, and brutally honest. Avoid generic warnings.

Respond with valid JSON:
{{
  "counter_thesis": "<2-3 sentence contrarian argument>",
  "key_vulnerability": "<the single biggest risk the consensus is underweighting>",
  "historical_analog": "<a historical situation where similar consensus was wrong>",
  "probability_consensus_wrong": <0.0-1.0>,
  "alternative_scenario": "<what happens if the contrarian view is correct>",
  "recommended_hedge": "<specific hedge or risk mitigation>"
}}"""

MODEL_DISPLAY_NAMES = {
    "grok": "grok-4",
    "gpt": "gpt-5.2-pro",
    "claude": "claude-opus-4",
}


def _get_all_llms() -> List[Dict[str, Any]]:
    llms = []

    xai_key = os.environ.get("XAI_API_KEY", "")
    if xai_key:
        from langchain_openai import ChatOpenAI
        llms.append({
            "name": "grok",
            "provider": "xAI",
            "model": "grok-4-fast-reasoning",
            "llm": ChatOpenAI(
                model="grok-4-fast-reasoning",
                temperature=0.1,
                api_key=xai_key,
                base_url="https://api.x.ai/v1",
            ),
        })

    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if openai_key:
        llms.append({
            "name": "gpt",
            "provider": "OpenAI",
            "model": "gpt-5.2-pro",
            "api_key": openai_key,
            "use_responses_api": True,
        })

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if anthropic_key:
        try:
            from langchain_anthropic import ChatAnthropic
            llms.append({
                "name": "claude",
                "provider": "Anthropic",
                "model": "claude-opus-4-6",
                "llm": ChatAnthropic(
                    model="claude-opus-4-6",
                    temperature=0.1,
                    api_key=anthropic_key,
                ),
            })
        except Exception as e:
            logger.warning(f"Anthropic init failed: {e}")

    return llms


def _extract_json(content: str) -> dict:
    import re
    content = content.strip()
    if content.startswith("```"):
        parts = content.split("```")
        if len(parts) >= 2:
            block = parts[1]
            if block.startswith("json"):
                block = block[4:]
            return json.loads(block.strip())
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{[\s\S]*\}', content)
    if match:
        return json.loads(match.group())
    raise json.JSONDecodeError("No JSON found", content, 0)


async def _gpt5_responses_call(api_key: str, prompt: str, system: str) -> str:
    async with httpx.AsyncClient(timeout=180) as client:
        resp = await client.post(
            "https://api.openai.com/v1/responses",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "gpt-5.2-pro",
                "instructions": system,
                "input": prompt,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        for item in data.get("output", []):
            if item.get("type") == "message":
                for c in item.get("content", []):
                    if c.get("type") == "output_text":
                        return c.get("text", "")
        return json.dumps(data)


async def _llm_call(llm_info: dict, prompt: str, system: str) -> dict:
    try:
        if llm_info.get("use_responses_api"):
            content = await _gpt5_responses_call(llm_info["api_key"], prompt, system)
            content = content.strip()
        else:
            from langchain_core.messages import HumanMessage, SystemMessage
            response = await llm_info["llm"].ainvoke([
                SystemMessage(content=system),
                HumanMessage(content=prompt),
            ])
            content = response.content.strip()
        parsed = _extract_json(content)
        parsed["_source"] = llm_info["name"]
        return parsed
    except json.JSONDecodeError:
        return {
            "_source": llm_info["name"],
            "raw_response": content if 'content' in dir() else "Parse error"
        }
    except Exception as e:
        logger.warning(f"LLM {llm_info['name']} failed: {e}")
        return {"_source": llm_info["name"], "error": str(e)}


def _detect_sector(symbol: str, fundamentals: dict = None) -> str:
    sector_map = {
        "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology", "GOOGL": "Communication Services",
        "AMZN": "Consumer Cyclical", "META": "Communication Services", "TSLA": "Consumer Cyclical",
        "AMD": "Technology", "NFLX": "Communication Services", "CRM": "Technology",
        "JPM": "Financial Services", "V": "Financial Services", "MA": "Financial Services",
        "BAC": "Financial Services", "GS": "Financial Services", "UNH": "Healthcare",
        "JNJ": "Healthcare", "PFE": "Healthcare", "ABBV": "Healthcare", "LLY": "Healthcare",
        "XOM": "Energy", "CVX": "Energy", "DIS": "Communication Services",
        "AVGO": "Technology", "PLTR": "Technology", "CRWD": "Technology",
        "COIN": "Financial Services", "SOFI": "Financial Services",
    }
    if symbol in sector_map:
        return sector_map[symbol]
    try:
        import yfinance as yf
        info = yf.Ticker(symbol).info
        return info.get("sector", "Other")
    except Exception:
        return "Other"


def _get_sector_template(sector: str) -> dict:
    for key, template in SECTOR_ANALYSIS_TEMPLATES.items():
        if key.lower() in sector.lower() or sector.lower() in key.lower():
            return template
    return DEFAULT_SECTOR_TEMPLATE


def _fetch_macro_regime_context() -> dict:
    try:
        from prediction.macro_intel import get_macro_dashboard
        macro = get_macro_dashboard()
        if macro and "error" not in macro:
            indicators = macro.get("indicators", [])
            regime_data = {}
            for ind in indicators:
                name = ind.get("name", "")
                val = ind.get("value")
                if val is not None and val != 0:
                    regime_data[name] = val
            return {
                "available_indicators": regime_data,
                "market_regime": macro.get("market_regime", "unknown"),
            }
    except Exception as e:
        logger.debug(f"Macro regime fetch failed: {e}")
    return {}


def disagreement_analysis(responses: Dict[str, dict]) -> dict:
    valid = {k: v for k, v in responses.items() if "error" not in v and "raw_response" not in v}
    if len(valid) < 2:
        return {"areas_of_agreement": [], "areas_of_disagreement": [], "consensus_strength": "weak"}

    areas_of_agreement = []
    areas_of_disagreement = []
    all_vals = list(valid.values())
    model_names = list(valid.keys())

    direction_fields = ["action", "sector_outlook", "fair_value_assessment", "time_horizon"]
    for field in direction_fields:
        values = [r.get(field) for r in all_vals if field in r]
        if not values:
            continue
        unique = set(str(v) for v in values)
        if len(unique) == 1:
            areas_of_agreement.append({"field": field, "value": values[0], "models": model_names})
        else:
            areas_of_disagreement.append({
                "field": field,
                "values": {MODEL_DISPLAY_NAMES.get(model_names[i], model_names[i]): str(values[i]) for i in range(len(values)) if i < len(values)},
                "category": "direction" if field == "action" else "outlook",
            })

    magnitude_fields = ["fundamental_score", "conviction", "entry_price", "stop_loss", "take_profit"]
    for field in magnitude_fields:
        values = [r.get(field) for r in all_vals if isinstance(r.get(field), (int, float))]
        if len(values) < 2:
            continue
        spread = max(values) - min(values)
        avg = sum(values) / len(values)
        threshold = avg * 0.15 if avg != 0 else 5
        if spread <= threshold:
            areas_of_agreement.append({"field": field, "avg_value": round(avg, 2), "spread": round(spread, 2)})
        else:
            areas_of_disagreement.append({
                "field": field,
                "values": {MODEL_DISPLAY_NAMES.get(model_names[i], model_names[i]): values[i] for i in range(len(values)) if i < len(values)},
                "spread": round(spread, 2),
                "category": "magnitude",
            })

    total_fields = len(areas_of_agreement) + len(areas_of_disagreement)
    if total_fields == 0:
        strength = "weak"
    else:
        agreement_ratio = len(areas_of_agreement) / total_fields
        if agreement_ratio >= 0.7:
            strength = "strong"
        elif agreement_ratio >= 0.4:
            strength = "moderate"
        else:
            strength = "weak"

    return {
        "areas_of_agreement": areas_of_agreement,
        "areas_of_disagreement": areas_of_disagreement,
        "consensus_strength": strength,
    }


def contrarian_check(llm_consensus_action: str, ml_recommendation: str, ml_direction: str) -> dict:
    buy_signals = {"STRONG_BUY", "BUY"}
    sell_signals = {"STRONG_SELL", "SELL"}
    ml_buy = ml_recommendation in buy_signals or ml_direction == "up"
    ml_sell = ml_recommendation in sell_signals or ml_direction == "down"
    llm_buy = llm_consensus_action in buy_signals
    llm_sell = llm_consensus_action in sell_signals

    is_contrarian = False
    explanation = "LLM consensus and ML prediction are aligned."

    if llm_buy and ml_sell:
        is_contrarian = True
        explanation = (
            f"CONTRARIAN SIGNAL: LLM consensus says {llm_consensus_action} but ML models predict "
            f"downside ({ml_recommendation}/{ml_direction}). The fundamental/qualitative view diverges "
            f"from quantitative technical signals. Exercise extra caution."
        )
    elif llm_sell and ml_buy:
        is_contrarian = True
        explanation = (
            f"CONTRARIAN SIGNAL: LLM consensus says {llm_consensus_action} but ML models predict "
            f"upside ({ml_recommendation}/{ml_direction}). Qualitative concerns override positive "
            f"technical momentum. Investigate fundamental risks."
        )
    elif llm_consensus_action == "HOLD" and (ml_buy or ml_sell):
        explanation = (
            f"LLM consensus is neutral (HOLD) while ML leans {ml_direction}. "
            f"Mixed signals suggest waiting for clearer confirmation."
        )

    return {"is_contrarian": is_contrarian, "explanation": explanation}


def _calculate_risk_reward_ratio(synthesis: dict, current_price: float) -> float:
    entry = synthesis.get("entry_price")
    stop_loss = synthesis.get("stop_loss")
    take_profit = synthesis.get("take_profit")

    if not all(isinstance(v, (int, float)) for v in [entry, stop_loss, take_profit]):
        entry = entry or current_price
        if not isinstance(stop_loss, (int, float)) or not isinstance(take_profit, (int, float)):
            return 0.0

    if not isinstance(entry, (int, float)):
        entry = current_price

    downside = abs(entry - stop_loss) if stop_loss else 0
    upside = abs(take_profit - entry) if take_profit else 0

    if downside == 0:
        return 0.0

    return round(upside / downside, 2)


def _compute_model_convictions(responses: Dict[str, dict], consensus_action: str) -> dict:
    action_alignment = {
        "STRONG_BUY": {"STRONG_BUY": 10, "BUY": 7, "HOLD": 4, "SELL": 2, "STRONG_SELL": 1},
        "BUY": {"STRONG_BUY": 8, "BUY": 10, "HOLD": 5, "SELL": 2, "STRONG_SELL": 1},
        "HOLD": {"STRONG_BUY": 3, "BUY": 5, "HOLD": 10, "SELL": 5, "STRONG_SELL": 3},
        "SELL": {"STRONG_BUY": 1, "BUY": 2, "HOLD": 5, "SELL": 10, "STRONG_SELL": 8},
        "STRONG_SELL": {"STRONG_BUY": 1, "BUY": 2, "HOLD": 4, "SELL": 7, "STRONG_SELL": 10},
    }

    model_convictions = {}
    alignment_map = action_alignment.get(consensus_action, {})

    for name, resp in responses.items():
        if "error" in resp or "raw_response" in resp:
            continue
        model_action = resp.get("action", "HOLD")
        model_conviction = resp.get("conviction", 5)
        if not isinstance(model_conviction, (int, float)):
            model_conviction = 5

        alignment_score = alignment_map.get(model_action, 5)
        final_score = round((alignment_score * 0.6 + min(model_conviction, 10) * 0.4))
        final_score = max(1, min(10, final_score))

        display_name = MODEL_DISPLAY_NAMES.get(name, name)
        model_convictions[display_name] = final_score

    return model_convictions


def _compute_weighted_consensus(responses: Dict[str, dict], model_track_records: dict) -> dict:
    valid = {k: v for k, v in responses.items() if "error" not in v and "raw_response" not in v}
    if not valid:
        return {}

    weights = {}
    for name in valid:
        display = MODEL_DISPLAY_NAMES.get(name, name)
        record = model_track_records.get(display, {})
        accuracy = record.get("accuracy_5d", 50) / 100.0
        sample_size = record.get("total_signals", 10)
        confidence_factor = min(1.0, sample_size / 50)
        weights[name] = max(0.1, accuracy * confidence_factor + (1 - confidence_factor) * 0.5)

    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {k: v / total_weight for k, v in weights.items()}
    else:
        weights = {k: 1.0 / len(valid) for k in valid}

    weighted_consensus = {}
    all_vals = list(valid.values())

    for key in all_vals[0]:
        if key.startswith("_"):
            continue
        values = [(valid[name].get(key), weights.get(name, 0)) for name in valid if key in valid[name]]
        if not values:
            continue

        vals = [v for v, w in values]
        wts = [w for v, w in values]

        if all(isinstance(v, (int, float)) for v in vals):
            weighted_consensus[key] = round(sum(v * w for v, w in zip(vals, wts)), 2)
        elif all(isinstance(v, str) for v in vals):
            from collections import Counter
            weighted_votes = Counter()
            for v, w in zip(vals, wts):
                weighted_votes[v] += w
            weighted_consensus[key] = weighted_votes.most_common(1)[0][0]
        elif all(isinstance(v, list) for v in vals):
            seen = set()
            merged = []
            for lst in vals:
                for item in lst:
                    item_str = str(item)
                    if item_str not in seen:
                        seen.add(item_str)
                        merged.append(item)
            weighted_consensus[key] = merged
        else:
            weighted_consensus[key] = vals[0]

    weighted_consensus["model_weights_used"] = {
        MODEL_DISPLAY_NAMES.get(k, k): round(v, 3) for k, v in weights.items()
    }

    return weighted_consensus


async def _multi_llm_analyze(prompt: str, system: str) -> Dict[str, Any]:
    llms = _get_all_llms()
    if not llms:
        return {"error": "No LLM API keys configured", "responses": {}}

    tasks = [_llm_call(llm, prompt, system) for llm in llms]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    responses = {}
    for i, r in enumerate(results):
        name = llms[i]["name"]
        if isinstance(r, Exception):
            responses[name] = {"error": str(r)}
        else:
            responses[name] = r

    valid_responses = {k: v for k, v in responses.items() if "error" not in v and "raw_response" not in v}

    if not valid_responses:
        first_valid = next((v for v in responses.values() if "error" not in v), None)
        if first_valid:
            return {"consensus": first_valid, "responses": responses, "agreement": 0}
        return {"error": "All LLMs failed", "responses": responses, "models_used": [], "models_succeeded": []}

    consensus = _build_consensus(valid_responses)
    consensus["responses"] = responses
    consensus["models_used"] = [llms[i]["name"] for i in range(len(llms))]
    consensus["models_succeeded"] = list(valid_responses.keys())

    return consensus


async def _run_contrarian_analysis(consensus_summary: str, symbol: str) -> dict:
    llms = _get_all_llms()
    if not llms:
        return {}

    contrarian_llm = llms[-1] if len(llms) > 1 else llms[0]

    prompt = f"""Consensus analysis for {symbol}:
{consensus_summary}

Now construct the strongest possible counter-argument against this consensus."""

    try:
        result = await _llm_call(contrarian_llm, prompt, CONTRARIAN_PROMPT)
        if "error" not in result and "raw_response" not in result:
            result.pop("_source", None)
            return result
    except Exception as e:
        logger.debug(f"Contrarian analysis failed: {e}")

    return {}


def _build_consensus(responses: Dict[str, dict]) -> dict:
    if len(responses) == 1:
        single = list(responses.values())[0]
        single["agreement"] = 1.0
        return single

    consensus = {}
    all_values = list(responses.values())

    for key in all_values[0]:
        if key.startswith("_"):
            continue
        values = [r.get(key) for r in all_values if key in r]
        if not values:
            continue

        if all(isinstance(v, (int, float)) for v in values):
            consensus[key] = round(sum(values) / len(values), 2)
        elif all(isinstance(v, str) for v in values):
            from collections import Counter
            counts = Counter(values)
            consensus[key] = counts.most_common(1)[0][0]
        elif all(isinstance(v, list) for v in values):
            seen = set()
            merged = []
            for lst in values:
                for item in lst:
                    item_str = str(item)
                    if item_str not in seen:
                        seen.add(item_str)
                        merged.append(item)
            consensus[key] = merged
        else:
            consensus[key] = values[0]

    actions = [r.get("action") for r in all_values if "action" in r]
    if actions:
        from collections import Counter
        action_counts = Counter(actions)
        consensus["action"] = action_counts.most_common(1)[0][0]
        consensus["action_agreement"] = round(action_counts.most_common(1)[0][1] / len(actions), 2)

    scores = [r.get("fundamental_score") for r in all_values if isinstance(r.get("fundamental_score"), (int, float))]
    if scores:
        consensus["score_spread"] = round(max(scores) - min(scores), 1)

    convictions = [r.get("conviction") for r in all_values if isinstance(r.get("conviction"), (int, float))]
    if convictions:
        consensus["conviction_spread"] = round(max(convictions) - min(convictions), 1)

    agreement_keys = ["action", "fair_value_assessment", "sector_outlook"]
    agreements = []
    for key in agreement_keys:
        vals = [r.get(key) for r in all_values if key in r]
        if vals:
            from collections import Counter
            c = Counter(vals)
            agreements.append(c.most_common(1)[0][1] / len(vals))
    consensus["agreement"] = round(sum(agreements) / len(agreements), 2) if agreements else 0

    return consensus


async def run_agentic_analysis(symbol: str, user_risk_tolerance: str = "moderate") -> Dict[str, Any]:
    from prediction.engine import get_engine

    steps_completed = []
    errors = []

    sector = _detect_sector(symbol)
    sector_template = _get_sector_template(sector)

    steps_completed.append({"agent": "data_collection", "status": "running"})
    ml_data = {}
    market_data = {}
    try:
        engine = get_engine()
        ml_data = engine.predict(symbol)
        if "error" not in ml_data:
            market_data = {
                "price": ml_data.get("price", 0),
                "change_1d": ml_data.get("change_1d", 0),
                "change_5d": ml_data.get("change_5d", 0),
                "volatility": ml_data.get("volatility_ann", 0),
                "rsi": ml_data.get("technicals", {}).get("rsi"),
                "macd_hist": ml_data.get("technicals", {}).get("macd_hist"),
                "adx": ml_data.get("technicals", {}).get("adx"),
                "bb_pctb": ml_data.get("technicals", {}).get("bb_pctb"),
                "stoch_k": ml_data.get("technicals", {}).get("stoch_k"),
                "williams_r": ml_data.get("technicals", {}).get("williams_r"),
                "confidence": ml_data.get("confidence", 0),
                "direction": ml_data.get("direction", "neutral"),
                "trend_pct": ml_data.get("predicted_trend_pct", 0),
                "recommendation": ml_data.get("recommendation", "HOLD"),
                "shap_factors": ml_data.get("shap_factors", []),
                "model_agreement": ml_data.get("model_agreement", 0),
                "regime": ml_data.get("regime", "neutral"),
                "edge_score": ml_data.get("edge_score", 0),
                "sharpe_estimate": ml_data.get("sharpe_estimate", 0),
            }
            steps_completed[-1]["status"] = "success"
        else:
            steps_completed[-1]["status"] = "partial"
            errors.append(ml_data["error"])
    except Exception as e:
        steps_completed[-1]["status"] = "error"
        errors.append(str(e))

    macro_context = _fetch_macro_regime_context()

    steps_completed.append({"agent": "fundamental_analysis", "status": "running"})
    fundamental = {}
    consensus_meta = {}
    real_fundamentals = {}
    earnings_proximity = {}
    try:
        try:
            from prediction.fundamentals import get_fundamentals
            fund_data = await get_fundamentals(symbol)
            if fund_data and "error" not in fund_data:
                ratios = fund_data.get("ratios", {})
                real_fundamentals = {
                    "pe_ratio": ratios.get("pe_ratio"),
                    "pb_ratio": ratios.get("pb_ratio"),
                    "ps_ratio": ratios.get("ps_ratio"),
                    "roe": ratios.get("roe"),
                    "roa": ratios.get("roa"),
                    "debt_to_equity": ratios.get("debt_to_equity"),
                    "current_ratio": ratios.get("current_ratio"),
                    "revenue_growth_yoy": ratios.get("revenue_growth_yoy"),
                    "earnings_growth_yoy": ratios.get("earnings_growth_yoy"),
                    "free_cash_flow_yield": ratios.get("free_cash_flow_yield"),
                    "peg_ratio": ratios.get("peg_ratio"),
                    "ev_to_ebitda": ratios.get("ev_to_ebitda"),
                }
        except ImportError:
            import yfinance as yf
            info = yf.Ticker(symbol).info
            if info:
                real_fundamentals = {
                    "pe_ratio": info.get("trailingPE") or info.get("forwardPE"),
                    "pb_ratio": info.get("priceToBook"),
                    "ps_ratio": info.get("priceToSalesTrailing12Months"),
                    "roe": round(info.get("returnOnEquity", 0) * 100, 2) if info.get("returnOnEquity") else None,
                    "roa": round(info.get("returnOnAssets", 0) * 100, 2) if info.get("returnOnAssets") else None,
                    "debt_to_equity": round(info.get("debtToEquity", 0) / 100, 2) if info.get("debtToEquity") else None,
                    "current_ratio": info.get("currentRatio"),
                    "revenue_growth_yoy": round(info.get("revenueGrowth", 0) * 100, 2) if info.get("revenueGrowth") else None,
                    "earnings_growth_yoy": round(info.get("earningsGrowth", 0) * 100, 2) if info.get("earningsGrowth") else None,
                    "free_cash_flow_yield": round(info.get("freeCashflow", 0) / info.get("marketCap", 1) * 100, 2) if info.get("freeCashflow") and info.get("marketCap") else None,
                    "peg_ratio": info.get("pegRatio"),
                    "ev_to_ebitda": info.get("enterpriseToEbitda"),
                }
    except Exception as e:
        logger.debug(f"Fundamentals fetch failed for {symbol}: {e}")

    try:
        from prediction.news_feed import get_earnings_calendar_sync
        earnings_list = get_earnings_calendar_sync([symbol])
        if earnings_list:
            e = earnings_list[0]
            days_until = e.get("days_until", 999)
            earnings_proximity = {
                "date": e.get("earnings_date"),
                "days_until": days_until,
                "eps_estimate": e.get("eps_estimate"),
                "within_2_weeks": days_until <= 14,
                "warning": "EARNINGS IMMINENT" if days_until <= 14 else None,
            }
    except Exception as e:
        logger.debug(f"Earnings calendar failed for {symbol}: {e}")

    insider_data = {}
    try:
        from prediction.flow_tracker import get_smart_money_flow
        flow = get_smart_money_flow(symbol)
        if flow:
            insider = flow.get("insider_activity", {})
            insider_data = {
                "net_sentiment": insider.get("net_sentiment"),
                "buy_count": insider.get("buy_count", 0),
                "sell_count": insider.get("sell_count", 0),
                "score": insider.get("score"),
            }
    except Exception as e:
        logger.debug(f"Insider data failed for {symbol}: {e}")

    options_context = ""
    try:
        from prediction.options_flow import get_options_flow
        opts = get_options_flow(symbol)
        if opts and "error" not in opts:
            summary = opts.get("summary", {})
            pc_ratio = opts.get("put_call_ratio", {})
            max_pain = opts.get("max_pain", {})
            options_context = f"\nOptions Flow Data:\n- Put/Call Volume Ratio: {pc_ratio.get('volume', 'N/A')}\n- Flow Sentiment: {summary.get('flow_sentiment', 'N/A')}\n- Unusual Activity Count: {summary.get('unusual_count', 0)}\n- Bullish Signals: {summary.get('bullish_signals', 0)}, Bearish Signals: {summary.get('bearish_signals', 0)}\n- Max Pain: ${max_pain.get('max_pain_strike', 'N/A')} (Distance: {max_pain.get('distance_pct', 'N/A')}%)"
            if summary.get("iv_rank") is not None:
                options_context += f"\n- IV Rank: {summary.get('iv_rank', 'N/A')}%"
    except Exception as e:
        logger.debug(f"Options flow failed for {symbol}: {e}")

    try:
        fund_section = ""
        if real_fundamentals:
            fund_items = [f"- {k.replace('_', ' ').title()}: {v}" for k, v in real_fundamentals.items() if v is not None]
            if fund_items:
                fund_section = "\nReal Fundamental Data:\n" + "\n".join(fund_items)

        earnings_section = ""
        if earnings_proximity:
            earnings_section = f"\nEarnings Proximity:\n- Next Earnings: {earnings_proximity.get('date', 'N/A')}\n- Days Until: {earnings_proximity.get('days_until', 'N/A')}\n- EPS Estimate: {earnings_proximity.get('eps_estimate', 'N/A')}"
            if earnings_proximity.get("within_2_weeks"):
                earnings_section += "\n- *** EARNINGS WITHIN 2 WEEKS — FLAG ELEVATED EVENT RISK ***"

        insider_section = ""
        if insider_data:
            insider_section = f"\nInsider Activity (SEC Form 4):\n- Net Sentiment: {insider_data.get('net_sentiment', 'N/A')}\n- Buys: {insider_data.get('buy_count', 0)}, Sells: {insider_data.get('sell_count', 0)}\n- Insider Score: {insider_data.get('score', 'N/A')}"

        macro_section = ""
        if macro_context:
            macro_items = macro_context.get("available_indicators", {})
            if macro_items:
                macro_lines = [f"- {k}: {v}" for k, v in list(macro_items.items())[:8]]
                macro_section = f"\nMacro Environment:\n- Market Regime: {macro_context.get('market_regime', 'unknown')}\n" + "\n".join(macro_lines)

        sector_section = f"\nSector Analysis Framework ({sector}):\n- Key Metrics: {', '.join(sector_template['key_metrics'])}\n- Focus: {sector_template['focus']}\n- Valuation: {sector_template['valuation_method']}\n- Risk Focus: {sector_template['risk_focus']}"

        shap_section = ""
        shap_factors = market_data.get("shap_factors", [])
        if shap_factors:
            shap_items = [f"- {f['feature']}: {f['direction']} (impact: {f['impact']:.4f})" for f in shap_factors[:5]]
            shap_section = "\nML Key Drivers (SHAP Analysis):\n" + "\n".join(shap_items)

        prompt = f"""Analyze {symbol} ({sector} sector) for investment potential.

Market Data:
- Price: ${market_data.get('price', 'N/A')}
- 1D Change: {market_data.get('change_1d', 0):.2f}%
- 5D Change: {market_data.get('change_5d', 0):.2f}%
- Volatility (ann.): {market_data.get('volatility', 0):.1f}%
- RSI(14): {market_data.get('rsi', 'N/A')}
- ADX(14): {market_data.get('adx', 'N/A')}
- MACD Histogram: {market_data.get('macd_hist', 'N/A')}
- Bollinger %B: {market_data.get('bb_pctb', 'N/A')}
- Stochastic K: {market_data.get('stoch_k', 'N/A')}

ML Prediction (XGBoost+LightGBM+CatBoost ensemble):
- Direction: {market_data.get('direction', 'neutral')}
- Confidence: {market_data.get('confidence', 0):.1f}%
- 5-day trend: {market_data.get('trend_pct', 0):.2f}%
- Model agreement: {market_data.get('model_agreement', 0):.1%}
- Market regime: {market_data.get('regime', 'neutral')}
- Edge score: {market_data.get('edge_score', 0)}
{shap_section}
{fund_section}
{earnings_section}
{insider_section}
{options_context}
{macro_section}
{sector_section}

Use the sector-specific analysis framework provided. Structure your analysis through: Macro Environment, Company Fundamentals (using sector-specific metrics), Technical Analysis, Catalyst Assessment, and Risk Factors.

SCORING RUBRIC:
- 90-100: Exceptional — clear catalyst, strong momentum, deep value, minimal risk
- 70-89: Strong — favorable setup with identifiable edge
- 50-69: Neutral — balanced risk/reward, no clear edge
- 30-49: Weak — deteriorating fundamentals or technicals
- 0-29: Avoid — significant breakdown

Provide a JSON response with:
{{
  "fundamental_score": <0-100>,
  "catalysts": ["<specific catalyst with timing>", "<catalyst2>", "<catalyst3>"],
  "risks": ["<specific quantified risk>", "<risk2>"],
  "sector_outlook": "positive" | "neutral" | "negative",
  "fair_value_assessment": "undervalued" | "fairly_valued" | "overvalued",
  "conviction": <1-10>,
  "key_insight": "<2-3 sentence insight with specific data points>",
  "sector_specific_notes": "<analysis using sector framework>"
}}"""

        result = await _multi_llm_analyze(
            prompt,
            STRUCTURED_REASONING_PROMPT
        )
        if "error" in result and not result.get("models_succeeded"):
            fundamental = result
            steps_completed[-1]["status"] = "partial"
            errors.append(f"Fundamental: {result['error']}")
        else:
            da = disagreement_analysis(result.get("responses", {}))
            consensus_meta["fundamental"] = {
                "models_used": result.get("models_used", []),
                "models_succeeded": result.get("models_succeeded", []),
                "agreement": result.get("agreement", 0),
                "individual_responses": {
                    k: {"score": v.get("fundamental_score"), "outlook": v.get("sector_outlook")}
                    for k, v in result.get("responses", {}).items()
                    if "error" not in v and "raw_response" not in v
                },
                "disagreement_analysis": da,
            }
            fundamental = {k: v for k, v in result.items() if k not in ("responses", "models_used", "models_succeeded")}
            steps_completed[-1]["status"] = "success"
    except Exception as e:
        steps_completed[-1]["status"] = "error"
        errors.append(str(e))

    steps_completed.append({"agent": "risk_assessment", "status": "running"})
    vol = market_data.get("volatility", 30) / 100
    trend = market_data.get("trend_pct", 0)
    confidence = market_data.get("confidence", 50)
    fund_score = fundamental.get("fundamental_score", 50)
    if not isinstance(fund_score, (int, float)):
        fund_score = 50
    var_95 = round(vol * 1.645 * 100, 2)
    max_loss = round(vol * 2.33 * 100, 2)
    win_prob = ml_data.get("probability", 0.5) if isinstance(ml_data, dict) else 0.5
    kelly = max(0, (win_prob * abs(trend) - (1 - win_prob) * var_95) / max(abs(trend), 0.1))
    position_size = round(min(kelly * 100, 25), 1)

    rsi_val = market_data.get("rsi")
    risk_score = round(
        (1 - confidence / 100) * 30 +
        min(vol * 100, 30) +
        (100 - fund_score) * 0.2 +
        (20 if rsi_val and (rsi_val > 70 or rsi_val < 30) else 0),
        1
    )
    risk_level = "LOW" if risk_score < 30 else "MEDIUM" if risk_score < 60 else "HIGH"

    risk_assessment = {
        "risk_score": min(round(risk_score, 1), 100),
        "var_95_pct": var_95,
        "max_loss_pct": max_loss,
        "position_size_pct": position_size,
        "risk_level": risk_level,
        "stop_loss_pct": round(var_95 * 1.5, 2),
        "take_profit_pct": round(abs(trend) * 2, 2),
    }
    steps_completed[-1]["status"] = "success"

    steps_completed.append({"agent": "compliance", "status": "running"})
    risk_map = {"conservative": 1, "moderate": 2, "aggressive": 3}
    user_risk_score = risk_map.get(user_risk_tolerance, 2)
    asset_risk_score = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}.get(risk_level, 2)
    suitable = user_risk_score >= asset_risk_score

    compliance = {
        "mifid_suitable": suitable,
        "user_risk_score": user_risk_score,
        "asset_risk_score": asset_risk_score,
        "eu_ai_act": {
            "system_type": "High-risk AI for financial decision support",
            "model_info": "Ensemble XGBoost + LightGBM + CatBoost with Optuna optimization",
            "llm_consensus": "Multi-LLM consensus (Grok-4 + GPT-5.2 Pro + Claude Opus 4)",
            "data_sources": "Yahoo Finance OHLCV, 94+ technical indicators",
            "human_oversight_required": True,
        },
        "disclaimer": (
            "FOR INFORMATIONAL PURPOSES ONLY. Not investment advice. "
            "AI predictions are probabilistic. Past performance ≠ future results. Capital at risk."
        ),
    }
    steps_completed[-1]["status"] = "success"

    steps_completed.append({"agent": "synthesis", "status": "running"})
    synthesis = {}
    try:
        synthesis_prompt = f"""Synthesize a final investment recommendation for {symbol} ({sector}).

ML Prediction (XGB+LGB+CatBoost): direction={market_data.get('direction')}, confidence={market_data.get('confidence')}%, trend={market_data.get('trend_pct')}%, model_agreement={market_data.get('model_agreement')}, regime={market_data.get('regime')}
Fundamental Analysis: {json.dumps(fundamental, default=str)}
Risk Assessment: {json.dumps(risk_assessment, default=str)}
MiFID Suitable: {suitable}
Current Price: ${market_data.get('price', 0)}
Sector: {sector}
Macro Regime: {macro_context.get('market_regime', 'unknown')}

Key SHAP Drivers: {json.dumps(market_data.get('shap_factors', [])[:3], default=str)}

Structure your reasoning through: Signal Alignment, Macro Context, Conviction Calibration, Position Sizing Logic, Scenario Analysis, and Final Verdict.

Provide JSON:
{{
  "action": "STRONG_BUY" | "BUY" | "HOLD" | "SELL" | "STRONG_SELL",
  "conviction": <1-10>,
  "reasoning": "<3-4 sentence synthesis referencing specific data points>",
  "entry_price": <suggested entry>,
  "stop_loss": <stop loss price>,
  "take_profit": <take profit price>,
  "time_horizon": "1-5 days" | "1-4 weeks" | "1-3 months",
  "key_risk": "<main risk with quantified impact>",
  "scenario_analysis": {{
    "bull_case": "<+2σ outcome with probability>",
    "base_case": "<expected outcome>",
    "bear_case": "<-2σ outcome with probability>"
  }}
}}"""

        result = await _multi_llm_analyze(
            synthesis_prompt,
            SYNTHESIS_REASONING_PROMPT
        )
        if "error" in result and not result.get("models_succeeded"):
            synthesis = result
            steps_completed[-1]["status"] = "partial"
            errors.append(f"Synthesis: {result['error']}")
        else:
            synth_da = disagreement_analysis(result.get("responses", {}))
            consensus_action = result.get("action", "HOLD")
            model_convictions = _compute_model_convictions(result.get("responses", {}), consensus_action)

            model_track_records = {}
            try:
                from prediction.signal_tracker import get_signal_performance
                perf = get_signal_performance()
                for model_name in MODEL_DISPLAY_NAMES.values():
                    model_track_records[model_name] = perf.get("by_model", {}).get(model_name, {})
            except Exception:
                pass

            weighted = _compute_weighted_consensus(result.get("responses", {}), model_track_records)

            consensus_meta["synthesis"] = {
                "models_used": result.get("models_used", []),
                "models_succeeded": result.get("models_succeeded", []),
                "agreement": result.get("agreement", 0),
                "action_agreement": result.get("action_agreement", 0),
                "individual_responses": {
                    k: {"action": v.get("action"), "conviction": v.get("conviction")}
                    for k, v in result.get("responses", {}).items()
                    if "error" not in v and "raw_response" not in v
                },
                "disagreement_analysis": synth_da,
                "model_convictions": model_convictions,
                "weighted_consensus": weighted.get("model_weights_used", {}),
            }
            synthesis = {k: v for k, v in result.items() if k not in ("responses", "models_used", "models_succeeded")}

            if weighted.get("action"):
                synthesis["weighted_action"] = weighted["action"]
            if weighted.get("conviction"):
                synthesis["weighted_conviction"] = weighted["conviction"]

            current_price = market_data.get("price", 0)
            rr_ratio = _calculate_risk_reward_ratio(synthesis, current_price)
            synthesis["risk_reward_ratio"] = rr_ratio

            steps_completed[-1]["status"] = "success"
    except Exception as e:
        steps_completed[-1]["status"] = "error"
        errors.append(str(e))

    steps_completed.append({"agent": "contrarian_analysis", "status": "running"})
    contrarian_analysis = {}
    try:
        consensus_summary = f"Action: {synthesis.get('action', 'HOLD')}, Conviction: {synthesis.get('conviction', 5)}/10, Reasoning: {synthesis.get('reasoning', 'N/A')}, Fundamental Score: {fundamental.get('fundamental_score', 'N/A')}, ML Direction: {market_data.get('direction', 'neutral')}, Key Risk: {synthesis.get('key_risk', 'N/A')}"
        contrarian_analysis = await _run_contrarian_analysis(consensus_summary, symbol)
        steps_completed[-1]["status"] = "success" if contrarian_analysis else "skipped"
    except Exception as e:
        steps_completed[-1]["status"] = "error"
        errors.append(f"Contrarian: {str(e)}")

    ml_recommendation = market_data.get("recommendation", "HOLD")
    ml_direction = market_data.get("direction", "neutral")
    llm_action = synthesis.get("action", "HOLD")
    contrarian = contrarian_check(llm_action, ml_recommendation, ml_direction)

    conviction_calibration = {}
    try:
        from prediction.signal_tracker import get_signal_performance
        perf = get_signal_performance()
        signal_stats = perf.get("by_type", {})
        llm_stats = signal_stats.get("agentic_analysis", signal_stats.get("composite_score", {}))
        historical_accuracy = llm_stats.get("accuracy_5d")
        if historical_accuracy is not None:
            llm_conviction = synthesis.get("conviction", 5)
            if not isinstance(llm_conviction, (int, float)):
                llm_conviction = 5
            calibration_factor = historical_accuracy / 100.0 if historical_accuracy > 0 else 0.5
            calibrated_conviction = round(llm_conviction * calibration_factor, 1)
            conviction_calibration = {
                "raw_conviction": llm_conviction,
                "historical_accuracy_5d": historical_accuracy,
                "calibration_factor": round(calibration_factor, 3),
                "calibrated_conviction": max(1, min(10, calibrated_conviction)),
                "note": f"LLM conviction {llm_conviction}/10 calibrated by {historical_accuracy:.1f}% historical accuracy",
            }
    except Exception as e:
        logger.debug(f"Conviction calibration failed: {e}")

    return {
        "symbol": symbol,
        "price": market_data.get("price", 0),
        "sector": sector,
        "ml_prediction": {
            "direction": market_data.get("direction", "neutral"),
            "confidence": market_data.get("confidence", 0),
            "trend_pct": market_data.get("trend_pct", 0),
            "recommendation": market_data.get("recommendation", "HOLD"),
            "shap_factors": market_data.get("shap_factors", []),
            "model_probas": ml_data.get("model_probas", {}) if isinstance(ml_data, dict) else {},
            "model_agreement": market_data.get("model_agreement", 0),
            "regime": market_data.get("regime", "neutral"),
            "edge_score": market_data.get("edge_score", 0),
        },
        "fundamental_analysis": fundamental,
        "real_fundamentals": real_fundamentals,
        "earnings_proximity": earnings_proximity,
        "insider_data": insider_data,
        "risk_assessment": risk_assessment,
        "compliance": compliance,
        "synthesis": synthesis,
        "consensus": consensus_meta,
        "conviction_calibration": conviction_calibration,
        "contrarian_check": contrarian,
        "contrarian_analysis": contrarian_analysis,
        "scenario_analysis": synthesis.get("scenario_analysis", {}),
        "macro_context": macro_context,
        "sector_analysis": {
            "sector": sector,
            "template_used": sector_template.get("focus", ""),
            "sector_specific_notes": fundamental.get("sector_specific_notes", ""),
        },
        "forecast": ml_data.get("forecast", {}) if isinstance(ml_data, dict) else {},
        "technicals": ml_data.get("technicals", {}) if isinstance(ml_data, dict) else {},
        "steps": steps_completed,
        "errors": errors,
        "timestamp": datetime.now().isoformat(),
    }
