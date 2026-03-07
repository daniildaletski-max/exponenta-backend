import asyncio
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger(__name__)

INTENT_PATTERNS = {
    "portfolio_query": [
        r"portfolio", r"my holdings", r"my stocks", r"my positions",
        r"how.*my.*doing", r"total value", r"p&l", r"pnl", r"profit",
        r"loss", r"allocation", r"diversif", r"rebalance", r"risk.*portfolio",
        r"sharpe", r"sortino", r"beta.*portfolio", r"drawdown",
    ],
    "ticker_analysis": [
        r"should i (buy|sell)", r"what.*think.*about", r"analyze\s+\$?[A-Z]{1,5}",
        r"score.*\$?[A-Z]{1,5}", r"thesis.*\$?[A-Z]{1,5}", r"target.*price",
        r"is\s+\$?[A-Z]{1,5}\s+(a )?(good|bad)", r"fair value",
        r"entry.*point", r"stop.*loss", r"take.*profit",
    ],
    "market_overview": [
        r"market", r"how.*market", r"indices", r"s&p", r"nasdaq",
        r"dow", r"breadth", r"fear.*greed", r"bull.*bear",
        r"sector.*perform", r"top.*gainer", r"top.*loser", r"movers",
    ],
    "strategy_question": [
        r"momentum", r"mean.*reversion", r"best.*plays", r"opportunities",
        r"scanner", r"screener", r"strong.*buy", r"setup",
        r"backtest", r"strategy", r"trend.*follow",
    ],
    "comparison": [
        r"compare", r"vs\.?", r"versus", r"better.*than",
        r"which.*is.*better", r"difference.*between",
    ],
    "general": [],
}

TICKER_PATTERN = re.compile(r'\$?([A-Z]{1,5})(?:\s|$|[,.\?!])')

_conversation_history: Dict[str, List[Dict[str, str]]] = {}
_HISTORY_MAX = 5


def _classify_intent(message: str) -> str:
    msg_lower = message.lower()
    scores = {}
    for intent, patterns in INTENT_PATTERNS.items():
        if intent == "general":
            continue
        score = sum(1 for p in patterns if re.search(p, msg_lower))
        if score > 0:
            scores[intent] = score
    if not scores:
        return "general"
    return max(scores, key=scores.get)


def _extract_tickers(message: str) -> List[str]:
    found = TICKER_PATTERN.findall(message.upper())
    known_tickers = {
        "AAPL", "TSLA", "NVDA", "GOOGL", "MSFT", "AMZN", "META", "VOO",
        "AMD", "NFLX", "JPM", "V", "BA", "DIS", "GOOG", "BRK", "SPY",
        "QQQ", "INTC", "CRM", "ORCL", "UBER", "COIN", "PLTR", "SOFI",
        "RIVN", "NIO", "PYPL", "SQ", "SHOP", "SNOW", "NET", "DDOG",
        "CRWD", "ZS", "PANW", "ABNB", "DASH", "RBLX", "U", "HOOD",
    }
    stop_words = {
        "I", "A", "THE", "IS", "IT", "MY", "AN", "OR", "AND", "DO", "IF", "SO", "AM", "BE",
        "TO", "IN", "ON", "AT", "OF", "VS", "BY", "UP", "NO", "AS", "HE", "WE", "ME", "US",
        "WHAT", "ARE", "TOP", "HOW", "WHY", "WHO", "CAN", "FOR", "NOT", "BUT", "ALL", "HAS",
        "HAD", "WAS", "HIS", "HER", "ITS", "OUR", "OUT", "GET", "GOT", "SET", "LET", "PUT",
        "SAY", "MAY", "DAY", "WAY", "ANY", "NEW", "OLD", "BIG", "LOW", "HIGH", "NOW", "RUN",
        "BUY", "SELL", "RISK", "BEST", "GOOD", "BAD", "MOST", "MAKE", "TAKE", "GIVE", "TELL",
        "SHOW", "FIND", "KEEP", "HELP", "LOOK", "LIKE", "WANT", "NEED", "KNOW", "THINK",
        "MUCH", "MANY", "SOME", "THIS", "THAT", "WITH", "FROM", "HAVE", "BEEN", "WILL",
        "THAN", "MORE", "ALSO", "JUST", "WHEN", "THEM", "THEN", "OVER", "ONLY", "VERY",
        "WELL", "BACK", "EVEN", "LONG", "ABOUT", "THEIR", "WHICH", "WOULD", "COULD", "SHOULD",
        "THESE", "THOSE", "AFTER", "BEFORE", "OTHER", "BEING", "THERE", "WHERE", "STILL",
        "CHECK", "WATCH", "TRADE", "STOCK", "STOCKS", "MONEY", "PRICE", "MARKET",
        "PORTFOLIO", "ANALYSIS", "OPPORTUNITIES", "SUMMARY", "SCORE", "ANALYZE",
        "ALYZE", "TRACK", "SIGNAL", "CHART", "TREND", "SECTOR", "INDEX",
    }
    result = []
    for t in found:
        if t in stop_words:
            continue
        if t in known_tickers:
            if t not in result:
                result.append(t)
        elif len(t) >= 2 and len(t) <= 5 and t not in stop_words:
            if t not in result:
                result.append(t)
    return result[:5]


async def _fetch_live_ticker_data(ticker: str) -> Dict[str, Any]:
    data = {}
    try:
        import data.market_data as md
        q = md._fetch_quote(ticker)
        if q:
            data["live_quote"] = {
                "price": q.get("price"),
                "change_pct": q.get("change_pct"),
                "volume": q.get("volume"),
            }
    except Exception as e:
        logger.debug(f"Live quote failed for {ticker}: {e}")

    try:
        try:
            from prediction.fundamentals import get_fundamentals
            fund = await get_fundamentals(ticker)
            if fund and "error" not in fund:
                ratios = fund.get("ratios", {})
                data["fundamentals"] = {
                    "pe_ratio": ratios.get("pe_ratio"),
                    "pb_ratio": ratios.get("pb_ratio"),
                    "ps_ratio": ratios.get("ps_ratio"),
                    "roe": ratios.get("roe"),
                    "debt_to_equity": ratios.get("debt_to_equity"),
                    "revenue_growth_yoy": ratios.get("revenue_growth_yoy"),
                    "earnings_growth_yoy": ratios.get("earnings_growth_yoy"),
                    "free_cash_flow_yield": ratios.get("free_cash_flow_yield"),
                    "market_cap": fund.get("overview", {}).get("market_cap"),
                }
                estimates = fund.get("analyst_estimates", [])
                if estimates:
                    data["analyst_estimates"] = estimates[:3]
        except ImportError:
            import yfinance as yf
            info = yf.Ticker(ticker).info
            if info:
                data["fundamentals"] = {
                    "pe_ratio": info.get("trailingPE") or info.get("forwardPE"),
                    "pb_ratio": info.get("priceToBook"),
                    "ps_ratio": info.get("priceToSalesTrailing12Months"),
                    "roe": round(info.get("returnOnEquity", 0) * 100, 2) if info.get("returnOnEquity") else None,
                    "debt_to_equity": round(info.get("debtToEquity", 0) / 100, 2) if info.get("debtToEquity") else None,
                    "revenue_growth_yoy": round(info.get("revenueGrowth", 0) * 100, 2) if info.get("revenueGrowth") else None,
                    "earnings_growth_yoy": round(info.get("earningsGrowth", 0) * 100, 2) if info.get("earningsGrowth") else None,
                    "free_cash_flow_yield": round(info.get("freeCashflow", 0) / info.get("marketCap", 1) * 100, 2) if info.get("freeCashflow") and info.get("marketCap") else None,
                    "market_cap": info.get("marketCap"),
                }
                if info.get("targetMeanPrice"):
                    data["analyst_estimates"] = [{
                        "period": "12mo target",
                        "eps_estimate": info.get("forwardEps"),
                        "target_price": info.get("targetMeanPrice"),
                        "recommendation": info.get("recommendationKey"),
                    }]
    except Exception as e:
        logger.debug(f"Fundamentals failed for {ticker}: {e}")

    try:
        from prediction.flow_tracker import get_smart_money_flow
        flow = get_smart_money_flow(ticker)
        if flow:
            insider = flow.get("insider_activity", {})
            institutional = flow.get("institutional_sentiment", {})
            data["insider_activity"] = {
                "net_sentiment": insider.get("net_sentiment"),
                "buy_count": insider.get("buy_count"),
                "sell_count": insider.get("sell_count"),
                "score": insider.get("score"),
            }
            data["institutional_flow"] = {
                "top_holders": institutional.get("top_holders", [])[:3],
                "net_sentiment": institutional.get("net_sentiment"),
            }
    except Exception as e:
        logger.debug(f"Smart money flow failed for {ticker}: {e}")

    try:
        from prediction.options_flow import get_options_flow
        opts = get_options_flow(ticker)
        if opts:
            data["options_flow"] = {
                "put_call_ratio": opts.get("summary", {}).get("put_call_ratio"),
                "max_pain": opts.get("max_pain", {}).get("strike"),
                "sentiment": opts.get("summary", {}).get("sentiment"),
                "unusual_count": len(opts.get("unusual_activity", [])),
            }
    except Exception as e:
        logger.debug(f"Options flow failed for {ticker}: {e}")

    try:
        from prediction.news_feed import get_earnings_calendar
        earnings = await get_earnings_calendar([ticker])
        if earnings:
            e = earnings[0]
            data["next_earnings"] = {
                "date": e.get("earnings_date"),
                "days_until": e.get("days_until"),
                "eps_estimate": e.get("eps_estimate"),
            }
    except Exception as e:
        logger.debug(f"Earnings calendar failed for {ticker}: {e}")

    return data


async def _gather_context(intent: str, tickers: List[str], portfolio_data: dict) -> Dict[str, Any]:
    context = {}

    if portfolio_data and portfolio_data.get("has_portfolio"):
        context["portfolio"] = portfolio_data

    if intent in ("ticker_analysis", "comparison") and tickers:
        context["ticker_data"] = {}
        for ticker in tickers[:3]:
            ticker_info = {}
            try:
                from prediction.engine import get_engine
                engine = get_engine()
                pred = engine.predict(ticker)
                if "error" not in pred:
                    ticker_info["ml_prediction"] = {
                        "price": pred.get("price"),
                        "direction": pred.get("direction"),
                        "confidence": pred.get("confidence"),
                        "predicted_trend_pct": pred.get("predicted_trend_pct"),
                        "recommendation": pred.get("recommendation"),
                        "risk_level": pred.get("risk_level"),
                        "edge_score": pred.get("edge_score"),
                    }
            except Exception as e:
                logger.warning(f"ML prediction failed for {ticker}: {e}")

            try:
                from prediction.composite_intel import get_composite_score
                score = await get_composite_score(ticker)
                ticker_info["composite_score"] = {
                    "score": score.get("composite_score"),
                    "direction": score.get("direction"),
                    "signal_label": score.get("signal_label"),
                    "confidence": score.get("confidence"),
                    "key_drivers": score.get("key_drivers", []),
                }
            except Exception as e:
                logger.warning(f"Composite score failed for {ticker}: {e}")

            live = await _fetch_live_ticker_data(ticker)
            ticker_info.update(live)

            context["ticker_data"][ticker] = ticker_info

    elif tickers:
        context["ticker_data"] = {}
        for ticker in tickers[:3]:
            live = await _fetch_live_ticker_data(ticker)
            context["ticker_data"][ticker] = live

    if intent == "market_overview":
        try:
            import data.market_data as md
            context["market"] = {
                "indices": md.get_real_indices(),
                "movers": md.get_real_movers(),
            }
            quotes = md.get_real_quotes()
            advancing = sum(1 for q in quotes if q["change_pct"] > 0)
            declining = sum(1 for q in quotes if q["change_pct"] < 0)
            avg_change = sum(q["change_pct"] for q in quotes) / len(quotes) if quotes else 0
            context["market"]["breadth"] = {
                "advancing": advancing,
                "declining": declining,
                "avg_change": round(avg_change, 2),
                "regime": "BULLISH" if avg_change > 0.5 else "BEARISH" if avg_change < -0.5 else "MIXED",
            }
        except Exception as e:
            logger.warning(f"Market overview failed: {e}")

    if intent == "strategy_question":
        try:
            from prediction.scanner import scan_universe
            scan = scan_universe(universe_keys=["us_mega_cap"], max_assets=5, use_ml=False)
            opps = scan.get("opportunities", [])
            context["top_opportunities"] = [
                {
                    "symbol": o.get("symbol"),
                    "score": o.get("composite_score"),
                    "signal": o.get("signal"),
                    "price": o.get("price"),
                    "change_pct": o.get("change_pct"),
                }
                for o in opps[:5]
            ]
        except Exception as e:
            logger.warning(f"Scanner failed: {e}")

    if portfolio_data and portfolio_data.get("has_portfolio") and tickers:
        holdings = portfolio_data.get("holdings", [])
        held_tickers = {h["ticker"] for h in holdings}
        relevant = [h for h in holdings if h["ticker"] in set(tickers)]
        if relevant:
            context["portfolio_relevance"] = relevant

    return context


def _build_system_prompt() -> str:
    return """You are Exponenta AI, an elite financial intelligence assistant embedded in the Exponenta investment terminal.

Your capabilities:
- Analyze individual stocks using ML predictions, composite scores, technical indicators, and sentiment
- Access real-time fundamentals: PE, PB, ROE, revenue growth, earnings growth, FCF yield, debt ratios
- Access real insider trading data (SEC Form 4 filings) and institutional holdings (13F filings)
- Access options flow intelligence: put/call ratios, max pain, unusual activity, IV rank
- Access analyst consensus estimates with EPS/revenue forecasts
- Review portfolio risk metrics, allocation, and diversification
- Provide market overview with breadth, indices, and sector performance
- Identify momentum opportunities and trade setups
- Generate trade theses with entry/exit levels
- Portfolio-aware responses: automatically reference user's holdings when relevant

Rules:
1. Be concise but thorough. Use bullet points and structure.
2. Always cite the data source (ML model, composite score, fundamentals, insider data, options flow, etc.)
3. Include specific numbers and metrics when available.
4. For buy/sell opinions, always mention risk level and confidence.
5. End actionable responses with a brief risk disclaimer.
6. Use markdown formatting for readability.
7. When referencing scores, use format like "Composite Score: 72/100"
8. If data is unavailable, say so honestly rather than guessing.
9. When the user holds the stock being discussed, mention their position and P&L.
10. When fundamental data is available, incorporate PE, revenue growth, and valuation into your analysis.
11. When insider/institutional data is available, highlight notable buying/selling patterns.
12. When options flow data is available, note unusual activity or extreme put/call ratios.

Personality: Professional, data-driven, slightly assertive. Think Bloomberg terminal meets conversational AI."""


def _build_user_prompt(message: str, intent: str, context: Dict[str, Any], history: List[Dict[str, str]]) -> str:
    parts = [f"User question: {message}\n"]
    parts.append(f"Detected intent: {intent}\n")

    if history:
        parts.append("Recent conversation context:")
        for h in history[-3:]:
            parts.append(f"  {h['role']}: {h['content'][:200]}")
        parts.append("")

    if "portfolio" in context and context["portfolio"]:
        p = context["portfolio"]
        if p.get("has_portfolio"):
            parts.append(f"Portfolio: Total value ${p.get('total_value', 0):,.2f}")
            holdings = p.get("holdings", [])
            if holdings:
                parts.append("Holdings:")
                for h in holdings[:10]:
                    parts.append(f"  {h['ticker']}: {h['quantity']} shares @ ${h.get('current_price', 0):.2f}, "
                                f"P&L: {h.get('pnl_pct', 0):+.1f}%, Weight: {h.get('weight', 0)*100:.1f}%")
        else:
            parts.append("Portfolio: No holdings configured yet.")

    if "ticker_data" in context:
        for ticker, data in context["ticker_data"].items():
            parts.append(f"\n{ticker} Analysis Data:")
            if "live_quote" in data:
                lq = data["live_quote"]
                parts.append(f"  Live Price: ${lq.get('price', 'N/A')}, Change: {lq.get('change_pct', 0):+.2f}%")
            if "ml_prediction" in data:
                ml = data["ml_prediction"]
                parts.append(f"  ML Prediction: {ml.get('direction', 'N/A')}, "
                           f"Confidence: {ml.get('confidence', 0):.1f}%, "
                           f"5d Trend: {ml.get('predicted_trend_pct', 0):+.2f}%, "
                           f"Signal: {ml.get('recommendation', 'N/A')}, "
                           f"Risk: {ml.get('risk_level', 'N/A')}")
            if "composite_score" in data:
                cs = data["composite_score"]
                parts.append(f"  Composite Score: {cs.get('score', 'N/A')}/100 ({cs.get('signal_label', 'N/A')}), "
                           f"Direction: {cs.get('direction', 'N/A')}, "
                           f"Confidence: {cs.get('confidence', 'N/A')}")
                drivers = cs.get("key_drivers", [])
                if drivers:
                    parts.append(f"  Key Drivers: {', '.join(d.get('signal', '') + '=' + str(d.get('score', '')) for d in drivers[:3])}")
            if "fundamentals" in data:
                f = data["fundamentals"]
                fund_parts = []
                if f.get("pe_ratio") is not None:
                    fund_parts.append(f"PE={f['pe_ratio']:.1f}")
                if f.get("pb_ratio") is not None:
                    fund_parts.append(f"PB={f['pb_ratio']:.1f}")
                if f.get("roe") is not None:
                    fund_parts.append(f"ROE={f['roe']:.1f}%")
                if f.get("debt_to_equity") is not None:
                    fund_parts.append(f"D/E={f['debt_to_equity']:.2f}")
                if f.get("revenue_growth_yoy") is not None:
                    fund_parts.append(f"RevGrowth={f['revenue_growth_yoy']:+.1f}%")
                if f.get("free_cash_flow_yield") is not None:
                    fund_parts.append(f"FCF Yield={f['free_cash_flow_yield']:.1f}%")
                if fund_parts:
                    parts.append(f"  Fundamentals: {', '.join(fund_parts)}")
            if "analyst_estimates" in data:
                ests = data["analyst_estimates"]
                est_strs = [f"{e.get('period', '?')}: EPS est ${e.get('eps_estimate', '?')}" for e in ests]
                parts.append(f"  Analyst Estimates: {'; '.join(est_strs)}")
            if "insider_activity" in data:
                ins = data["insider_activity"]
                parts.append(f"  Insider Activity: Buys={ins.get('buy_count', 0)}, Sells={ins.get('sell_count', 0)}, "
                           f"Sentiment={ins.get('net_sentiment', 'N/A')}, Score={ins.get('score', 'N/A')}")
            if "institutional_flow" in data:
                inst = data["institutional_flow"]
                holders = inst.get("top_holders", [])
                if holders:
                    holder_str = ", ".join(h.get("name", "?") for h in holders[:3])
                    parts.append(f"  Top Institutional Holders: {holder_str}")
                parts.append(f"  Institutional Sentiment: {inst.get('net_sentiment', 'N/A')}")
            if "options_flow" in data:
                opt = data["options_flow"]
                parts.append(f"  Options Flow: P/C Ratio={opt.get('put_call_ratio', 'N/A')}, "
                           f"Max Pain=${opt.get('max_pain', 'N/A')}, "
                           f"Sentiment={opt.get('sentiment', 'N/A')}, "
                           f"Unusual Activity={opt.get('unusual_count', 0)} signals")
            if "next_earnings" in data:
                ne = data["next_earnings"]
                parts.append(f"  Next Earnings: {ne.get('date', 'N/A')} ({ne.get('days_until', '?')} days away), "
                           f"EPS Estimate: ${ne.get('eps_estimate', 'N/A')}")

    if "market" in context:
        m = context["market"]
        indices = m.get("indices", [])
        if indices:
            parts.append("\nMarket Indices:")
            for idx in indices:
                parts.append(f"  {idx['name']}: {idx.get('value', 0):,.2f} ({idx.get('change_pct', 0):+.2f}%)")
        breadth = m.get("breadth", {})
        if breadth:
            parts.append(f"  Regime: {breadth.get('regime')}, Advancing: {breadth.get('advancing')}, Declining: {breadth.get('declining')}")
        movers = m.get("movers", {})
        gainers = movers.get("gainers", [])
        losers = movers.get("losers", [])
        if gainers:
            parts.append(f"  Top Gainers: {', '.join(g['ticker'] + ' ' + str(g['change_pct']) + '%' for g in gainers[:3])}")
        if losers:
            parts.append(f"  Top Losers: {', '.join(l['ticker'] + ' ' + str(l['change_pct']) + '%' for l in losers[:3])}")

    if "top_opportunities" in context:
        parts.append("\nTop Scanned Opportunities:")
        for o in context["top_opportunities"]:
            parts.append(f"  {o['symbol']}: Score {o.get('score', 'N/A')}, Signal: {o.get('signal', 'N/A')}, "
                       f"Price: ${o.get('price', 0):.2f}, Change: {o.get('change_pct', 0):+.2f}%")

    if "portfolio_relevance" in context:
        parts.append("\nPortfolio Relevance (user holds these tickers):")
        for h in context["portfolio_relevance"]:
            parts.append(f"  {h['ticker']}: {h.get('quantity', 0)} shares, P&L: {h.get('pnl_pct', 0):+.1f}%, "
                       f"Weight: {h.get('weight', 0)*100:.1f}%")

    parts.append("\nProvide a helpful, data-driven response. Use markdown formatting.")
    return "\n".join(parts)


async def _call_llm_stream(system_prompt: str, user_prompt: str):
    xai_key = os.environ.get("XAI_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if xai_key:
        async for chunk in _stream_openai_compatible(
            api_key=xai_key,
            base_url="https://api.x.ai/v1",
            model="grok-4-fast-reasoning",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        ):
            yield chunk
        return

    if openai_key:
        async for chunk in _stream_openai_compatible(
            api_key=openai_key,
            base_url="https://api.openai.com/v1",
            model="gpt-5.2-pro",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        ):
            yield chunk
        return

    if anthropic_key:
        async for chunk in _stream_anthropic(
            api_key=anthropic_key,
            model="claude-opus-4-6",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        ):
            yield chunk
        return

    yield _generate_fallback_response(user_prompt)


async def _stream_openai_compatible(api_key: str, base_url: str, model: str, system_prompt: str, user_prompt: str):
    async with httpx.AsyncClient(timeout=120) as client:
        try:
            async with client.stream(
                "POST",
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": True,
                    "temperature": 0.3,
                    "max_tokens": 1500,
                },
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        parsed = json.loads(data)
                        delta = parsed.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except (json.JSONDecodeError, IndexError, KeyError):
                        continue
        except Exception as e:
            logger.error(f"LLM streaming error: {e}")
            yield f"I encountered an error connecting to the AI model. Here's what I can tell you based on the data:\n\n"
            yield _generate_fallback_response(user_prompt)


async def _stream_anthropic(api_key: str, model: str, system_prompt: str, user_prompt: str):
    async with httpx.AsyncClient(timeout=120) as client:
        try:
            async with client.stream(
                "POST",
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": model,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": user_prompt}],
                    "stream": True,
                    "max_tokens": 1500,
                    "temperature": 0.3,
                },
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    try:
                        parsed = json.loads(line[6:])
                        if parsed.get("type") == "content_block_delta":
                            text = parsed.get("delta", {}).get("text", "")
                            if text:
                                yield text
                    except (json.JSONDecodeError, KeyError):
                        continue
        except Exception as e:
            logger.error(f"Anthropic streaming error: {e}")
            yield _generate_fallback_response(user_prompt)


def _generate_fallback_response(user_prompt: str) -> str:
    lines = user_prompt.split("\n")
    data_lines = [l for l in lines if l.strip().startswith(("Portfolio:", "Holdings:", "ML Prediction:", "Composite Score:", "Market Indices:", "Top"))]
    if data_lines:
        return "Based on the available data:\n\n" + "\n".join(f"- {l.strip()}" for l in data_lines[:10]) + "\n\n*Configure an API key (XAI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY) for AI-powered analysis.*"
    return "I need an LLM API key to provide intelligent analysis. Please configure XAI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY in your environment."


def get_conversation_history(session_id: str) -> List[Dict[str, str]]:
    return _conversation_history.get(session_id, [])


def add_to_history(session_id: str, role: str, content: str):
    if session_id not in _conversation_history:
        _conversation_history[session_id] = []
    _conversation_history[session_id].append({"role": role, "content": content[:500]})
    if len(_conversation_history[session_id]) > _HISTORY_MAX * 2:
        _conversation_history[session_id] = _conversation_history[session_id][-_HISTORY_MAX * 2:]


async def process_chat_message(message: str, session_id: str = "default", portfolio_data: dict = None):
    intent = _classify_intent(message)
    tickers = _extract_tickers(message)

    history = get_conversation_history(session_id)

    if not tickers and history:
        for h in reversed(history):
            prev_tickers = _extract_tickers(h["content"])
            if prev_tickers:
                tickers = prev_tickers
                break

    add_to_history(session_id, "user", message)

    context = await _gather_context(intent, tickers, portfolio_data or {})

    system_prompt = _build_system_prompt()
    user_prompt = _build_user_prompt(message, intent, context, history)

    full_response = []

    yield json.dumps({"type": "meta", "intent": intent, "tickers": tickers}) + "\n"

    async for chunk in _call_llm_stream(system_prompt, user_prompt):
        full_response.append(chunk)
        yield json.dumps({"type": "chunk", "content": chunk}) + "\n"

    complete_response = "".join(full_response)
    add_to_history(session_id, "assistant", complete_response)

    yield json.dumps({"type": "done"}) + "\n"
