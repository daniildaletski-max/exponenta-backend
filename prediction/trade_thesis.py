import asyncio
import json
import logging
import math
import os
from datetime import datetime
from typing import Any, Dict

import httpx

logger = logging.getLogger(__name__)

SECTOR_THESIS_TEMPLATES = {
    "Technology": {
        "framework": "Growth + Innovation Moat",
        "key_drivers": ["TAM expansion", "AI/cloud revenue mix", "R&D efficiency", "developer ecosystem", "product-led growth"],
        "valuation_anchors": ["EV/Revenue vs growth", "Rule of 40", "FCF margin trajectory"],
        "risk_checklist": ["Regulatory (antitrust)", "concentration in key products", "talent retention", "capex cycle"],
    },
    "Financial Services": {
        "framework": "NIM + Credit Quality",
        "key_drivers": ["Net interest margin", "loan growth", "fee income", "credit quality", "capital returns"],
        "valuation_anchors": ["P/TBV vs ROE", "dividend yield", "efficiency ratio"],
        "risk_checklist": ["Credit cycle", "rate sensitivity", "regulatory capital", "CRE exposure"],
    },
    "Healthcare": {
        "framework": "Pipeline Optionality",
        "key_drivers": ["Pipeline catalysts (FDA dates)", "patent cliff timeline", "pricing dynamics", "M&A probability"],
        "valuation_anchors": ["Sum-of-parts NPV", "P/E with pipeline adjustment", "EV/EBITDA"],
        "risk_checklist": ["Binary FDA risk", "patent expiration", "pricing regulation", "clinical trial failure"],
    },
    "Energy": {
        "framework": "Capital Discipline + FCF",
        "key_drivers": ["Production growth", "breakeven price", "capital allocation", "reserve life", "energy transition"],
        "valuation_anchors": ["EV/EBITDA", "FCF yield", "NAV on proved reserves"],
        "risk_checklist": ["Commodity price", "ESG/transition risk", "geopolitical", "regulatory"],
    },
}

DEFAULT_THESIS_TEMPLATE = {
    "framework": "Fundamental Quality",
    "key_drivers": ["Revenue growth", "margin expansion", "competitive moat", "management quality"],
    "valuation_anchors": ["P/E relative", "EV/EBITDA", "DCF"],
    "risk_checklist": ["Competitive threats", "macro sensitivity", "execution risk"],
}


def _get_thesis_template(sector: str) -> dict:
    for key, template in SECTOR_THESIS_TEMPLATES.items():
        if key.lower() in sector.lower() or sector.lower() in key.lower():
            return template
    return DEFAULT_THESIS_TEMPLATE


def _kelly_criterion(win_prob: float, avg_win: float, avg_loss: float) -> float:
    if avg_loss == 0 or avg_win == 0:
        return 0.0
    b = avg_win / avg_loss
    kelly = (win_prob * b - (1 - win_prob)) / b
    return max(0.0, min(kelly, 0.25))


def _detect_sector(ticker: str) -> str:
    sector_map = {
        "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology", "GOOGL": "Communication Services",
        "AMZN": "Consumer Cyclical", "META": "Communication Services", "TSLA": "Consumer Cyclical",
        "AMD": "Technology", "NFLX": "Communication Services", "CRM": "Technology",
        "JPM": "Financial Services", "V": "Financial Services", "MA": "Financial Services",
        "BAC": "Financial Services", "GS": "Financial Services", "UNH": "Healthcare",
        "JNJ": "Healthcare", "PFE": "Healthcare", "ABBV": "Healthcare", "LLY": "Healthcare",
        "XOM": "Energy", "CVX": "Energy", "AVGO": "Technology",
    }
    if ticker in sector_map:
        return sector_map[ticker]
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        return info.get("sector", "Other")
    except Exception:
        return "Other"


def _build_thesis_inputs(ticker: str) -> Dict[str, Any]:
    inputs: Dict[str, Any] = {}

    try:
        from prediction.engine import get_engine
        engine = get_engine()
        ml = engine.predict(ticker)
        if "error" not in ml:
            inputs["ml"] = {
                "price": ml.get("price", 0),
                "direction": ml.get("direction", "neutral"),
                "confidence": ml.get("confidence", 0),
                "trend_pct": ml.get("predicted_trend_pct", 0),
                "recommendation": ml.get("recommendation", "HOLD"),
                "risk_level": ml.get("risk_level", "MEDIUM"),
                "volatility_ann": ml.get("volatility_ann", 0),
                "model_accuracy": ml.get("model_accuracy", 0),
                "model_agreement": ml.get("model_agreement", 0),
                "edge_score": ml.get("edge_score", 0),
                "sharpe_estimate": ml.get("sharpe_estimate", 0),
                "shap_factors": ml.get("shap_factors", [])[:5],
                "technicals": ml.get("technicals", {}),
                "forecast": ml.get("forecast", {}),
                "change_1d": ml.get("change_1d", 0),
                "change_5d": ml.get("change_5d", 0),
                "probability": ml.get("probability", 0.5),
                "regime": ml.get("regime", "neutral"),
            }
    except Exception as e:
        logger.warning(f"ML prediction failed for thesis: {e}")

    try:
        from prediction.features import fetch_ohlcv
        import numpy as np
        df = fetch_ohlcv(ticker, days=60)
        if df is not None and len(df) > 20:
            closes = df["close"].values
            volumes = df["volume"].values
            sma20 = float(np.mean(closes[-20:]))
            sma50 = float(np.mean(closes[-50:])) if len(closes) >= 50 else sma20
            avg_vol = float(np.mean(volumes[-20:]))
            atr_vals = []
            for i in range(1, min(15, len(df))):
                tr = max(
                    float(df["high"].iloc[i] - df["low"].iloc[i]),
                    abs(float(df["high"].iloc[i] - df["close"].iloc[i - 1])),
                    abs(float(df["low"].iloc[i] - df["close"].iloc[i - 1])),
                )
                atr_vals.append(tr)
            atr = float(np.mean(atr_vals)) if atr_vals else 0

            daily_returns = np.diff(closes) / closes[:-1]
            sharpe_realized = float(np.mean(daily_returns) / (np.std(daily_returns) + 1e-10) * np.sqrt(252))

            inputs["technicals_extra"] = {
                "sma20": round(sma20, 2),
                "sma50": round(sma50, 2),
                "avg_volume_20d": round(avg_vol),
                "atr_14": round(atr, 2),
                "above_sma20": closes[-1] > sma20,
                "above_sma50": closes[-1] > sma50,
                "price_52w_high": round(float(np.max(closes)), 2),
                "price_52w_low": round(float(np.min(closes)), 2),
                "realized_sharpe_20d": round(sharpe_realized, 2),
                "max_drawdown_60d": round(float(np.min(np.minimum.accumulate(closes) / np.maximum.accumulate(closes) - 1) * 100), 2),
            }
    except Exception as e:
        logger.warning(f"Technicals extra failed: {e}")

    try:
        from prediction.news_feed import get_earnings_calendar_sync
        earnings = get_earnings_calendar_sync([ticker])
        if earnings:
            inputs["earnings"] = earnings[:2]
            e = earnings[0]
            days_until = e.get("days_until", 999)
            inputs["earnings_proximity"] = {
                "date": e.get("earnings_date"),
                "days_until": days_until,
                "eps_estimate": e.get("eps_estimate"),
                "within_2_weeks": days_until <= 14,
            }
    except Exception:
        pass

    try:
        from prediction.flow_tracker import get_smart_money_flow
        flow = get_smart_money_flow(ticker)
        if flow:
            insider = flow.get("insider_activity", {})
            institutional = flow.get("institutional_sentiment", {})
            inputs["insider_activity"] = {
                "net_sentiment": insider.get("net_sentiment"),
                "buy_count": insider.get("buy_count", 0),
                "sell_count": insider.get("sell_count", 0),
                "score": insider.get("score"),
                "recent_transactions": insider.get("recent_transactions", [])[:5],
            }
            inputs["institutional_holdings"] = {
                "top_holders": institutional.get("top_holders", [])[:5],
                "net_sentiment": institutional.get("net_sentiment"),
            }
    except Exception:
        pass

    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        if info:
            inputs["fundamentals"] = {
                "pe_ratio": info.get("trailingPE") or info.get("forwardPE"),
                "forward_pe": info.get("forwardPE"),
                "pb_ratio": info.get("priceToBook"),
                "ps_ratio": info.get("priceToSalesTrailing12Months"),
                "roe": round(info.get("returnOnEquity", 0) * 100, 2) if info.get("returnOnEquity") else None,
                "debt_to_equity": round(info.get("debtToEquity", 0) / 100, 2) if info.get("debtToEquity") else None,
                "revenue_growth_yoy": round(info.get("revenueGrowth", 0) * 100, 2) if info.get("revenueGrowth") else None,
                "earnings_growth_yoy": round(info.get("earningsGrowth", 0) * 100, 2) if info.get("earningsGrowth") else None,
                "free_cash_flow_yield": round(info.get("freeCashflow", 0) / info.get("marketCap", 1) * 100, 2) if info.get("freeCashflow") and info.get("marketCap") else None,
                "current_ratio": info.get("currentRatio"),
                "market_cap_b": round(info.get("marketCap", 0) / 1e9, 1) if info.get("marketCap") else None,
                "beta": info.get("beta"),
                "dividend_yield": round(info.get("dividendYield", 0) * 100, 2) if info.get("dividendYield") else None,
                "peg_ratio": info.get("pegRatio"),
                "sector": info.get("sector", "Unknown"),
            }
            inputs["analyst_estimates"] = []
            if info.get("targetMeanPrice"):
                inputs["analyst_estimates"].append({
                    "period": "12mo target",
                    "eps_estimate": info.get("forwardEps"),
                    "revenue_estimate": info.get("revenueEstimate"),
                    "target_price": info.get("targetMeanPrice"),
                    "target_high": info.get("targetHighPrice"),
                    "target_low": info.get("targetLowPrice"),
                    "recommendation": info.get("recommendationKey"),
                    "num_analysts": info.get("numberOfAnalystOpinions"),
                })
    except Exception:
        pass

    try:
        from prediction.options_flow import get_options_flow
        opts = get_options_flow(ticker)
        if opts and "error" not in opts:
            summary = opts.get("summary", {})
            inputs["options_context"] = {
                "put_call_ratio": opts.get("put_call_ratio", {}).get("volume"),
                "flow_sentiment": summary.get("flow_sentiment"),
                "unusual_count": summary.get("unusual_count", 0),
                "bullish_signals": summary.get("bullish_signals", 0),
                "bearish_signals": summary.get("bearish_signals", 0),
                "max_pain": opts.get("max_pain", {}).get("max_pain_strike"),
                "iv_rank": summary.get("iv_rank"),
            }
    except Exception:
        pass

    return inputs


def _compute_position_sizing(inputs: Dict[str, Any]) -> Dict[str, Any]:
    ml = inputs.get("ml", {})
    price = ml.get("price", 100)
    trend_pct = ml.get("trend_pct", 0)
    confidence = ml.get("confidence", 50)
    volatility = ml.get("volatility_ann", 30)
    probability = ml.get("probability", 0.5)

    vol_daily = volatility / 100 / math.sqrt(252) if volatility > 0 else 0.02
    atr_pct = vol_daily * price * math.sqrt(5)

    if trend_pct > 0:
        entry = round(price * 0.995, 2)
        stop_loss = round(price * (1 - max(vol_daily * 3, 0.02)), 2)
        take_profit = round(price * (1 + abs(trend_pct) / 100 * 2), 2)
    else:
        entry = round(price * 1.005, 2)
        stop_loss = round(price * (1 + max(vol_daily * 3, 0.02)), 2)
        take_profit = round(price * (1 - abs(trend_pct) / 100 * 2), 2)

    risk_per_share = abs(entry - stop_loss)
    reward_per_share = abs(take_profit - entry)
    risk_reward = round(reward_per_share / risk_per_share, 2) if risk_per_share > 0 else 0

    avg_win = abs(trend_pct) / 100 if trend_pct != 0 else 0.02
    avg_loss = risk_per_share / price if price > 0 else 0.02
    kelly = _kelly_criterion(probability, avg_win, avg_loss)
    half_kelly = round(kelly * 0.5 * 100, 1)

    fixed_fractional = round(min(2.0 / (risk_per_share / price * 100), 25), 1) if risk_per_share > 0 and price > 0 else 5.0
    vol_adjusted = round(min(10.0 / (volatility / 100 * math.sqrt(252 / 5)), 25), 1) if volatility > 0 else 5.0
    recommended = round(min(half_kelly, fixed_fractional, vol_adjusted, 15.0), 1)

    return {
        "entry_price": entry,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "risk_reward_ratio": risk_reward,
        "kelly_full_pct": round(kelly * 100, 1),
        "kelly_half_pct": half_kelly,
        "fixed_fractional_pct": fixed_fractional,
        "volatility_adjusted_pct": vol_adjusted,
        "recommended_position_pct": recommended,
        "max_loss_per_trade_pct": round(risk_per_share / price * 100, 2) if price > 0 else 0,
        "sizing_method": "min(half-Kelly, fixed-fractional 2%, vol-target 10%)",
    }


async def _llm_generate_thesis(ticker: str, inputs: Dict[str, Any], sizing: Dict[str, Any]) -> Dict[str, Any]:
    ml = inputs.get("ml", {})
    technicals = ml.get("technicals", {})
    techn_extra = inputs.get("technicals_extra", {})
    earnings = inputs.get("earnings", [])
    shap = ml.get("shap_factors", [])
    insider = inputs.get("insider_activity", {})
    institutional = inputs.get("institutional_holdings", {})
    fundamentals = inputs.get("fundamentals", {})
    analyst_estimates = inputs.get("analyst_estimates", [])
    earnings_prox = inputs.get("earnings_proximity", {})
    options_ctx = inputs.get("options_context", {})

    sector = fundamentals.get("sector", _detect_sector(ticker))
    thesis_template = _get_thesis_template(sector)

    fund_section = ""
    if fundamentals:
        fund_items = [f"- {k.replace('_', ' ').title()}: {v}" for k, v in fundamentals.items() if v is not None]
        if fund_items:
            fund_section = "\nFUNDAMENTAL DATA:\n" + "\n".join(fund_items)

    insider_section = ""
    if insider:
        insider_section = f"\nINSIDER ACTIVITY (SEC Form 4):\n- Net Sentiment: {insider.get('net_sentiment', 'N/A')}\n- Buys: {insider.get('buy_count', 0)}, Sells: {insider.get('sell_count', 0)}\n- Insider Score: {insider.get('score', 'N/A')}"

    institutional_section = ""
    if institutional:
        holders = institutional.get("top_holders", [])
        if holders:
            holder_names = ", ".join(h.get("name", "?") for h in holders[:3])
            institutional_section = f"\nINSTITUTIONAL HOLDERS (13F):\n- Top Holders: {holder_names}\n- Institutional Sentiment: {institutional.get('net_sentiment', 'N/A')}"

    analyst_section = ""
    if analyst_estimates:
        est = analyst_estimates[0]
        analyst_section = f"\nANALYST CONSENSUS:\n- Target Price: ${est.get('target_price', '?')} (Range: ${est.get('target_low', '?')} - ${est.get('target_high', '?')})\n- Forward EPS: ${est.get('eps_estimate', '?')}\n- Recommendation: {est.get('recommendation', '?')}\n- Analyst Count: {est.get('num_analysts', '?')}"

    earnings_warning = ""
    if earnings_prox and earnings_prox.get("within_2_weeks"):
        earnings_warning = f"\n*** CATALYST ALERT: EARNINGS IN {earnings_prox.get('days_until', '?')} DAYS ({earnings_prox.get('date', 'N/A')}) — EPS Est: ${earnings_prox.get('eps_estimate', 'N/A')} ***"

    options_section = ""
    if options_ctx:
        options_section = f"\nOPTIONS FLOW INTELLIGENCE:\n- Put/Call Ratio: {options_ctx.get('put_call_ratio', 'N/A')}\n- Flow Sentiment: {options_ctx.get('flow_sentiment', 'N/A')}\n- Unusual Activity: {options_ctx.get('unusual_count', 0)} signals\n- Bullish/Bearish Signals: {options_ctx.get('bullish_signals', 0)}/{options_ctx.get('bearish_signals', 0)}\n- Max Pain: ${options_ctx.get('max_pain', 'N/A')}"
        if options_ctx.get("iv_rank") is not None:
            options_section += f"\n- IV Rank: {options_ctx.get('iv_rank')}%"

    sector_framework = f"\nSECTOR ANALYSIS FRAMEWORK ({sector} — {thesis_template['framework']}):\n- Key Drivers: {', '.join(thesis_template['key_drivers'])}\n- Valuation Anchors: {', '.join(thesis_template['valuation_anchors'])}\n- Risk Checklist: {', '.join(thesis_template['risk_checklist'])}"

    prompt = f"""Generate a comprehensive institutional-quality trade thesis for {ticker} ({sector}).

MARKET DATA:
- Current Price: ${ml.get('price', 'N/A')}
- 1D Change: {ml.get('change_1d', 0):.2f}%
- 5D Change: {ml.get('change_5d', 0):.2f}%
- Annualized Volatility: {ml.get('volatility_ann', 0):.1f}%
- Market Regime: {ml.get('regime', 'neutral')}

ML PREDICTION (XGBoost+LightGBM+CatBoost Ensemble):
- Direction: {ml.get('direction', 'neutral')}
- Confidence: {ml.get('confidence', 0):.1f}%
- 5-Day Trend: {ml.get('trend_pct', 0):.2f}%
- Model Accuracy: {ml.get('model_accuracy', 0):.1f}%
- Model Agreement: {ml.get('model_agreement', 0):.1%}
- Recommendation: {ml.get('recommendation', 'HOLD')}
- Edge Score: {ml.get('edge_score', 0)}

TECHNICAL INDICATORS:
- RSI(14): {technicals.get('rsi', 'N/A')}
- MACD Histogram: {technicals.get('macd_hist', 'N/A')}
- ADX: {technicals.get('adx', 'N/A')}
- Bollinger %B: {technicals.get('bb_pctb', 'N/A')}
- SMA20: {techn_extra.get('sma20', 'N/A')}, SMA50: {techn_extra.get('sma50', 'N/A')}
- Above SMA20: {techn_extra.get('above_sma20', 'N/A')}, Above SMA50: {techn_extra.get('above_sma50', 'N/A')}
- ATR(14): {techn_extra.get('atr_14', 'N/A')}
- Realized Sharpe (20d): {techn_extra.get('realized_sharpe_20d', 'N/A')}
- Max Drawdown (60d): {techn_extra.get('max_drawdown_60d', 'N/A')}%

KEY ML DRIVERS (SHAP): {json.dumps(shap, default=str)}
{fund_section}
{insider_section}
{institutional_section}
{analyst_section}
{options_section}
{earnings_warning}
{sector_framework}

POSITION SIZING:
- Entry: ${sizing.get('entry_price')}
- Stop Loss: ${sizing.get('stop_loss')}
- Take Profit: ${sizing.get('take_profit')}
- Risk/Reward: {sizing.get('risk_reward_ratio')}
- Kelly Criterion: {sizing.get('kelly_half_pct')}% (half-Kelly)
- Recommended Size: {sizing.get('recommended_position_pct')}% ({sizing.get('sizing_method')})

UPCOMING EARNINGS: {json.dumps(earnings, default=str) if earnings else 'None scheduled'}

Use the sector-specific framework to structure your analysis. Provide a JSON response:
{{
  "investment_thesis": {{
    "headline": "<one-line thesis with specific price target>",
    "bull_case": ["<specific data-driven point>", "<point2>", "<point3>"],
    "bear_case": ["<specific risk with quantification>", "<point2>", "<point3>"],
    "catalyst_timeline": ["<catalyst with specific date/timeframe>"],
    "sector_context": "<how sector dynamics affect this position>",
    "contrarian_view": "<strongest argument against this thesis>"
  }},
  "conviction_score": <1-10>,
  "conviction_rationale": "<why this conviction level, referencing specific metrics>",
  "time_horizon": "intraday" | "swing" | "position" | "long_term",
  "risk_factors": ["<specific quantified risk>", "<risk2>", "<risk3>"],
  "key_levels": {{
    "strong_support": <price>,
    "strong_resistance": <price>,
    "pivot_point": <price>,
    "breakout_level": <price>,
    "breakdown_level": <price>
  }},
  "scenario_analysis": {{
    "bull_target": <price>,
    "bull_probability": <0.0-1.0>,
    "base_target": <price>,
    "base_probability": <0.0-1.0>,
    "bear_target": <price>,
    "bear_probability": <0.0-1.0>
  }}
}}"""

    system = """You are an elite institutional equity research analyst at a top-tier investment bank. 
Generate research-quality trade theses with specific, actionable insights. Be precise about price levels, 
catalysts, and risk factors. Your analysis should be data-driven, referencing the ML signals, technicals, 
and fundamentals provided. Use the sector-specific framework to add domain expertise.
Structure your reasoning before providing JSON. Respond with valid JSON only after your analysis."""

    xai_key = os.environ.get("XAI_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")

    llm_result = {}

    if xai_key:
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage, SystemMessage
            llm = ChatOpenAI(
                model="grok-4-fast-reasoning",
                temperature=0.2,
                api_key=xai_key,
                base_url="https://api.x.ai/v1",
            )
            response = await llm.ainvoke([
                SystemMessage(content=system),
                HumanMessage(content=prompt),
            ])
            content = response.content.strip()
            llm_result = _extract_json(content)
            llm_result["_llm"] = "grok-4"
        except Exception as e:
            logger.warning(f"Grok thesis generation failed: {e}")

    if not llm_result and openai_key:
        try:
            async with httpx.AsyncClient(timeout=180) as client:
                resp = await client.post(
                    "https://api.openai.com/v1/responses",
                    headers={"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"},
                    json={
                        "model": "gpt-5.2-pro",
                        "instructions": system,
                        "input": prompt,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                text = ""
                for item in data.get("output", []):
                    if item.get("type") == "message":
                        for c in item.get("content", []):
                            if c.get("type") == "output_text":
                                text = c.get("text", "")
                if text:
                    llm_result = _extract_json(text)
                    llm_result["_llm"] = "gpt-5.2-pro"
        except Exception as e:
            logger.warning(f"GPT thesis generation failed: {e}")

    return llm_result


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
    return {}


def _build_fallback_thesis(ticker: str, inputs: Dict[str, Any], sizing: Dict[str, Any]) -> Dict[str, Any]:
    ml = inputs.get("ml", {})
    direction = ml.get("direction", "neutral")
    trend = ml.get("trend_pct", 0)
    confidence = ml.get("confidence", 50)
    rec = ml.get("recommendation", "HOLD")
    price = ml.get("price", 0)
    vol = ml.get("volatility_ann", 30)
    technicals = ml.get("technicals", {})
    rsi = technicals.get("rsi")

    if direction == "bullish":
        bull_case = [
            f"ML ensemble signals bullish with {confidence:.0f}% confidence",
            f"Predicted 5-day upside of {trend:.2f}%",
            f"Model agreement of {ml.get('model_agreement', 0):.0%} across XGBoost/LightGBM/CatBoost",
        ]
        bear_case = [
            f"Annualized volatility at {vol:.1f}% creates risk",
            "Broad market downturn could override positive signals",
            f"RSI at {rsi:.0f} — {'overbought territory' if rsi and rsi > 70 else 'monitor momentum'}" if rsi else "Monitor technical momentum shifts",
        ]
    elif direction == "bearish":
        bull_case = [
            "Mean reversion potential after decline",
            "Oversold conditions may attract buyers",
            "Long-term fundamentals may remain intact",
        ]
        bear_case = [
            f"ML ensemble signals bearish with {confidence:.0f}% confidence",
            f"Predicted 5-day downside of {trend:.2f}%",
            f"Elevated volatility at {vol:.1f}% annualized",
        ]
    else:
        bull_case = [
            "Market conditions may improve",
            f"Current price ${price:.2f} near support levels",
            "Sector tailwinds possible",
        ]
        bear_case = [
            "No clear directional signal from ML models",
            f"Mixed model agreement at {ml.get('model_agreement', 0):.0%}",
            "Wait for catalyst confirmation",
        ]

    conviction = min(10, max(1, int(confidence / 10)))

    return {
        "investment_thesis": {
            "headline": f"{ticker}: {'Bullish' if direction == 'bullish' else 'Bearish' if direction == 'bearish' else 'Neutral'} setup — ML ensemble {rec} signal",
            "bull_case": bull_case,
            "bear_case": bear_case,
            "catalyst_timeline": [
                "Earnings announcement (check calendar)",
                "Sector rotation dynamics in play",
            ],
            "sector_context": "Monitor sector-level flows for confirmation",
            "contrarian_view": "Consensus trades are crowded; watch for mean reversion if positioning becomes extreme",
        },
        "conviction_score": conviction,
        "conviction_rationale": f"Based on {confidence:.0f}% ML confidence and {ml.get('model_agreement', 0):.0%} model agreement",
        "time_horizon": "swing",
        "risk_factors": [
            f"Volatility risk — {vol:.1f}% annualized",
            "Model prediction uncertainty",
            "Macro event risk",
        ],
        "key_levels": {
            "strong_support": round(price * 0.95, 2),
            "strong_resistance": round(price * 1.05, 2),
            "pivot_point": round(price, 2),
            "breakout_level": round(price * 1.03, 2),
            "breakdown_level": round(price * 0.97, 2),
        },
        "scenario_analysis": {
            "bull_target": round(price * 1.08, 2),
            "bull_probability": 0.3,
            "base_target": round(price * (1 + trend / 100), 2),
            "base_probability": 0.5,
            "bear_target": round(price * 0.92, 2),
            "bear_probability": 0.2,
        },
    }


async def generate_trade_thesis(ticker: str) -> Dict[str, Any]:
    ticker = ticker.upper()

    inputs = _build_thesis_inputs(ticker)
    if not inputs.get("ml"):
        return {"error": f"Could not generate ML prediction for {ticker}"}

    ml = inputs["ml"]
    sizing = _compute_position_sizing(inputs)

    llm_thesis = await _llm_generate_thesis(ticker, inputs, sizing)

    if not llm_thesis or "investment_thesis" not in llm_thesis:
        llm_thesis = _build_fallback_thesis(ticker, inputs, sizing)

    llm_used = llm_thesis.pop("_llm", "fallback")

    thesis = llm_thesis.get("investment_thesis", {})
    conviction = llm_thesis.get("conviction_score", 5)
    if not isinstance(conviction, (int, float)):
        conviction = 5
    conviction = max(1, min(10, conviction))

    insider_summary = inputs.get("insider_activity", {})
    institutional_summary = inputs.get("institutional_holdings", {})
    fund_data = inputs.get("fundamentals", {})
    analyst_ests = inputs.get("analyst_estimates", [])
    earnings_prox = inputs.get("earnings_proximity", {})
    options_ctx = inputs.get("options_context", {})

    risk_quantification = {}
    if fund_data:
        de = fund_data.get("debt_to_equity")
        cr = fund_data.get("current_ratio")
        fcfy = fund_data.get("free_cash_flow_yield")
        risk_items = []
        if de is not None and de > 2.0:
            risk_items.append(f"High leverage (D/E={de:.2f})")
        if cr is not None and cr < 1.0:
            risk_items.append(f"Liquidity concern (Current Ratio={cr:.2f})")
        if fcfy is not None and fcfy < 0:
            risk_items.append(f"Negative free cash flow yield ({fcfy:.1f}%)")
        risk_quantification = {
            "debt_to_equity": de,
            "current_ratio": cr,
            "free_cash_flow_yield": fcfy,
            "flags": risk_items,
        }

    sector = fund_data.get("sector", _detect_sector(ticker))

    return {
        "ticker": ticker,
        "price": ml.get("price", 0),
        "sector": sector,
        "thesis": {
            "headline": thesis.get("headline", f"{ticker} Trade Thesis"),
            "bull_case": thesis.get("bull_case", []),
            "bear_case": thesis.get("bear_case", []),
            "catalyst_timeline": thesis.get("catalyst_timeline", []),
            "sector_context": thesis.get("sector_context", ""),
            "contrarian_view": thesis.get("contrarian_view", ""),
        },
        "conviction": {
            "score": conviction,
            "rationale": llm_thesis.get("conviction_rationale", ""),
            "time_horizon": llm_thesis.get("time_horizon", "swing"),
        },
        "entry_strategy": {
            "entry_price": sizing["entry_price"],
            "stop_loss": sizing["stop_loss"],
            "take_profit": sizing["take_profit"],
            "risk_reward_ratio": sizing["risk_reward_ratio"],
        },
        "position_sizing": {
            "kelly_full_pct": sizing["kelly_full_pct"],
            "kelly_half_pct": sizing["kelly_half_pct"],
            "fixed_fractional_pct": sizing["fixed_fractional_pct"],
            "volatility_adjusted_pct": sizing["volatility_adjusted_pct"],
            "recommended_pct": sizing["recommended_position_pct"],
            "max_loss_per_trade_pct": sizing["max_loss_per_trade_pct"],
            "sizing_method": sizing["sizing_method"],
        },
        "risk_factors": llm_thesis.get("risk_factors", []),
        "risk_quantification": risk_quantification,
        "key_levels": llm_thesis.get("key_levels", {}),
        "scenario_analysis": llm_thesis.get("scenario_analysis", {}),
        "ml_signal": {
            "direction": ml.get("direction", "neutral"),
            "confidence": ml.get("confidence", 0),
            "trend_pct": ml.get("trend_pct", 0),
            "recommendation": ml.get("recommendation", "HOLD"),
            "model_accuracy": ml.get("model_accuracy", 0),
            "model_agreement": ml.get("model_agreement", 0),
            "shap_factors": ml.get("shap_factors", []),
            "regime": ml.get("regime", "neutral"),
        },
        "insider_activity": insider_summary,
        "institutional_holdings": institutional_summary,
        "fundamentals": fund_data,
        "analyst_estimates": analyst_ests,
        "earnings_proximity": earnings_prox,
        "options_flow": options_ctx,
        "technicals": ml.get("technicals", {}),
        "forecast": ml.get("forecast", {}),
        "llm_used": llm_used,
        "generated_at": datetime.now().isoformat(),
    }
