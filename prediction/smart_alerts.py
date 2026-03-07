import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import yfinance as yf

from prediction.features import build_feature_matrix, fetch_ohlcv

logger = logging.getLogger(__name__)

PRIORITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


def _make_alert(
    alert_type: str,
    ticker: str,
    priority: str,
    title: str,
    message: str,
    data: Optional[dict] = None,
) -> dict:
    return {
        "id": str(uuid.uuid4())[:12],
        "type": alert_type,
        "ticker": ticker,
        "priority": priority,
        "title": title,
        "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data": data or {},
    }


class AlertEngine:

    def price_alerts(self, tickers: list[str], prices_data: dict) -> list[dict]:
        alerts = []
        for ticker in tickers:
            try:
                df = fetch_ohlcv(ticker, days=365)
                if df is None or len(df) < 10:
                    continue

                close = float(df["close"].iloc[-1])
                prev_close = float(df["close"].iloc[-2])
                change_pct = (close - prev_close) / prev_close * 100

                if abs(change_pct) > 2:
                    direction = "up" if change_pct > 0 else "down"
                    priority = "critical" if abs(change_pct) > 5 else "high"
                    alerts.append(_make_alert(
                        alert_type="price_move",
                        ticker=ticker,
                        priority=priority,
                        title=f"{ticker} {'surged' if direction == 'up' else 'dropped'} {abs(change_pct):.1f}%",
                        message=f"{ticker} moved {change_pct:+.2f}% from ${prev_close:.2f} to ${close:.2f}",
                        data={"change_pct": round(change_pct, 2), "price": round(close, 2), "prev_close": round(prev_close, 2)},
                    ))

                high_52w = float(np.nanmax(df["close"].values))
                low_52w = float(np.nanmin(df["close"].values))

                if high_52w > 0 and (high_52w - close) / high_52w <= 0.03:
                    alerts.append(_make_alert(
                        alert_type="52w_high",
                        ticker=ticker,
                        priority="high",
                        title=f"{ticker} near 52-week high",
                        message=f"{ticker} at ${close:.2f}, within 3% of 52-week high ${high_52w:.2f}",
                        data={"price": round(close, 2), "high_52w": round(high_52w, 2)},
                    ))

                if low_52w > 0 and (close - low_52w) / low_52w <= 0.03:
                    alerts.append(_make_alert(
                        alert_type="52w_low",
                        ticker=ticker,
                        priority="high",
                        title=f"{ticker} near 52-week low",
                        message=f"{ticker} at ${close:.2f}, within 3% of 52-week low ${low_52w:.2f}",
                        data={"price": round(close, 2), "low_52w": round(low_52w, 2)},
                    ))
            except Exception as e:
                logger.debug(f"Price alert error for {ticker}: {e}")
        return alerts

    def volume_alerts(self, tickers: list[str]) -> list[dict]:
        alerts = []
        for ticker in tickers:
            try:
                df = fetch_ohlcv(ticker, days=60)
                if df is None or len(df) < 21:
                    continue

                current_vol = float(df["volume"].iloc[-1])
                avg_vol_20 = float(df["volume"].iloc[-21:-1].mean())

                if avg_vol_20 > 0 and current_vol > 2 * avg_vol_20:
                    ratio = current_vol / avg_vol_20
                    priority = "critical" if ratio > 5 else "high" if ratio > 3 else "medium"
                    alerts.append(_make_alert(
                        alert_type="unusual_volume",
                        ticker=ticker,
                        priority=priority,
                        title=f"{ticker} unusual volume ({ratio:.1f}x avg)",
                        message=f"{ticker} trading {ratio:.1f}x its 20-day average volume ({current_vol:,.0f} vs avg {avg_vol_20:,.0f})",
                        data={"current_volume": current_vol, "avg_volume_20d": round(avg_vol_20), "ratio": round(ratio, 2)},
                    ))
            except Exception as e:
                logger.debug(f"Volume alert error for {ticker}: {e}")
        return alerts

    def earnings_alerts(self, tickers: list[str]) -> list[dict]:
        alerts = []
        for ticker in tickers:
            try:
                t = yf.Ticker(ticker)
                cal = t.calendar
                if cal is None:
                    continue

                earnings_date = None
                if isinstance(cal, dict):
                    ed = cal.get("Earnings Date")
                    if ed is not None:
                        if isinstance(ed, list) and len(ed) > 0:
                            earnings_date = ed[0]
                        else:
                            earnings_date = ed
                elif hasattr(cal, "iloc"):
                    try:
                        earnings_date = cal.iloc[0, 0]
                    except Exception:
                        pass

                if earnings_date is None:
                    continue

                import pandas as pd
                if isinstance(earnings_date, list):
                    earnings_date = earnings_date[0] if earnings_date else None
                if earnings_date is None:
                    continue
                earnings_ts = pd.Timestamp(str(earnings_date))
                if earnings_ts.tz is not None:
                    earnings_ts = earnings_ts.tz_localize(None)

                now = pd.Timestamp.now().normalize()
                earnings_ts = earnings_ts.normalize()

                days_until = (earnings_ts - now).days

                if 0 <= days_until <= 5:
                    priority = "critical" if days_until <= 1 else "high" if days_until <= 3 else "medium"
                    alerts.append(_make_alert(
                        alert_type="earnings_approaching",
                        ticker=ticker,
                        priority=priority,
                        title=f"{ticker} earnings in {days_until} day{'s' if days_until != 1 else ''}",
                        message=f"{ticker} reports earnings on {earnings_ts.strftime('%b %d, %Y')} ({days_until} trading days away)",
                        data={"days_until": days_until, "earnings_date": str(earnings_ts)[:10]},
                    ))
            except Exception as e:
                logger.debug(f"Earnings alert error for {ticker}: {e}")
        return alerts

    def sentiment_shift_alerts(self, tickers: list[str]) -> list[dict]:
        alerts = []
        for ticker in tickers:
            try:
                df = build_feature_matrix(ticker, days=100)
                if df is None or "rsi" not in df.columns:
                    continue

                rsi = df["rsi"].iloc[-1]
                if np.isnan(rsi):
                    continue

                if rsi > 70:
                    priority = "high" if rsi > 80 else "medium"
                    alerts.append(_make_alert(
                        alert_type="rsi_overbought",
                        ticker=ticker,
                        priority=priority,
                        title=f"{ticker} RSI overbought ({rsi:.0f})",
                        message=f"{ticker} RSI at {rsi:.1f} — potentially overbought, watch for reversal",
                        data={"rsi": round(float(rsi), 1)},
                    ))
                elif rsi < 30:
                    priority = "high" if rsi < 20 else "medium"
                    alerts.append(_make_alert(
                        alert_type="rsi_oversold",
                        ticker=ticker,
                        priority=priority,
                        title=f"{ticker} RSI oversold ({rsi:.0f})",
                        message=f"{ticker} RSI at {rsi:.1f} — potentially oversold, watch for bounce",
                        data={"rsi": round(float(rsi), 1)},
                    ))
            except Exception as e:
                logger.debug(f"Sentiment alert error for {ticker}: {e}")
        return alerts

    def opportunity_alerts(self, risk_tolerance: str = "moderate") -> list[dict]:
        alerts = []
        try:
            from prediction.scanner import scan_universe
            result = scan_universe(universe_keys=["us_mega_cap"], max_assets=10, use_ml=False, min_score=70)
            opportunities = result.get("opportunities", [])

            for opp in opportunities:
                if opp.get("signal") != "STRONG_BUY":
                    continue

                risk_level = opp.get("risk_level", "MEDIUM")
                if risk_tolerance == "conservative" and risk_level == "HIGH":
                    continue
                if risk_tolerance == "aggressive" or risk_tolerance == "moderate" or risk_level != "HIGH":
                    alerts.append(_make_alert(
                        alert_type="opportunity",
                        ticker=opp["symbol"],
                        priority="medium",
                        title=f"{opp['symbol']} STRONG BUY signal (score {opp['composite_score']})",
                        message=f"{opp['symbol']} at ${opp['price']:.2f} has a composite score of {opp['composite_score']} with {opp.get('technical_setup', 'N/A')} setup",
                        data={
                            "composite_score": opp["composite_score"],
                            "signal": opp["signal"],
                            "risk_level": risk_level,
                            "price": opp["price"],
                        },
                    ))
        except Exception as e:
            logger.debug(f"Opportunity alert error: {e}")
        return alerts


def get_all_alerts(tickers: list[str], risk_tolerance: str = "moderate") -> dict:
    engine = AlertEngine()
    all_alerts = []

    try:
        all_alerts.extend(engine.price_alerts(tickers, {}))
    except Exception as e:
        logger.error(f"Price alerts failed: {e}")

    try:
        all_alerts.extend(engine.volume_alerts(tickers))
    except Exception as e:
        logger.error(f"Volume alerts failed: {e}")

    try:
        all_alerts.extend(engine.earnings_alerts(tickers))
    except Exception as e:
        logger.error(f"Earnings alerts failed: {e}")

    try:
        all_alerts.extend(engine.sentiment_shift_alerts(tickers))
    except Exception as e:
        logger.error(f"Sentiment alerts failed: {e}")

    try:
        all_alerts.extend(engine.opportunity_alerts(risk_tolerance))
    except Exception as e:
        logger.error(f"Opportunity alerts failed: {e}")

    all_alerts.sort(key=lambda a: PRIORITY_ORDER.get(a["priority"], 3))

    critical_count = sum(1 for a in all_alerts if a["priority"] == "critical")

    return {
        "alerts": all_alerts,
        "count": len(all_alerts),
        "critical_count": critical_count,
    }
