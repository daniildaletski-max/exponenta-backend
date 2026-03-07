import os
import time
import logging
from typing import Any, Dict, Optional, List
from datetime import datetime, timezone

from prediction.api_utils import resilient_get
from prediction.cache_manager import SmartCache

logger = logging.getLogger(__name__)

FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "")
FMP_API_KEY = os.environ.get("FMP_API_KEY", "")
ALPHA_VANTAGE_KEY = os.environ.get("ALPHAVANTAGE_API_KEY", "")
FINNHUB_BASE = "https://finnhub.io/api/v1"
FMP_BASE = "https://financialmodelingprep.com/stable"
AV_BASE = "https://www.alphavantage.co/query"

_cache = SmartCache("fundamentals", max_size=500, default_ttl=600)
CACHE_TTL = 600


def _cached(key: str, ttl: int = CACHE_TTL) -> Optional[Any]:
    return _cache.get(key, ttl)


def _set_cache(key: str, val: Any):
    _cache.set(key, val)


def _finnhub_get(path: str, params: dict = None) -> Any:
    if not FINNHUB_API_KEY:
        return None
    p = params or {}
    p["token"] = FINNHUB_API_KEY
    try:
        r = resilient_get(f"{FINNHUB_BASE}{path}", params=p, timeout=15, source="finnhub")
        if r and r.status_code == 200:
            return r.json()
    except Exception as e:
        logger.warning(f"Finnhub request failed {path}: {e}")
    return None


def _av_get(function: str, params: dict = None) -> Any:
    if not ALPHA_VANTAGE_KEY:
        return None
    p = params or {}
    p["function"] = function
    p["apikey"] = ALPHA_VANTAGE_KEY
    try:
        r = resilient_get(AV_BASE, params=p, timeout=15, source="alpha_vantage")
        if r and r.status_code == 200:
            data = r.json()
            if "Error Message" not in data and "Note" not in data and "Information" not in data:
                return data
    except Exception as e:
        logger.warning(f"Alpha Vantage request failed {function}: {e}")
    return None


def _fmp_get(endpoint: str, params: dict = None) -> Any:
    if not FMP_API_KEY:
        return None
    p = params or {}
    p["apikey"] = FMP_API_KEY
    try:
        r = resilient_get(f"{FMP_BASE}/{endpoint}", params=p, timeout=15, source="fmp")
        if r is None:
            return None
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, dict) and ("Error Message" in data or "error" in data or "message" in data):
                logger.info(f"FMP {endpoint} returned error payload")
                return None
            return data
        if r.status_code in (402, 403):
            logger.info(f"FMP {endpoint} returned {r.status_code}")
            return None
    except Exception as e:
        logger.warning(f"FMP request failed {endpoint}: {e}")
    return None


def _yf_get_ticker(ticker: str):
    try:
        import yfinance as yf
        return yf.Ticker(ticker)
    except Exception:
        return None


def _safe_float(val, default=None):
    if val is None:
        return default
    try:
        f = float(val)
        if f != f:
            return default
        return f
    except (ValueError, TypeError):
        return default


def _format_large_number(val):
    if val is None:
        return None
    v = abs(val)
    sign = -1 if val < 0 else 1
    if v >= 1e12:
        return f"${sign * v / 1e12:.2f}T"
    if v >= 1e9:
        return f"${sign * v / 1e9:.2f}B"
    if v >= 1e6:
        return f"${sign * v / 1e6:.2f}M"
    return f"${sign * v:,.0f}"


def _parse_fmp_income(items: list, limit: int = 8) -> list:
    result = []
    for item in items[:limit]:
        revenue = _safe_float(item.get("revenue"))
        gross = _safe_float(item.get("grossProfit"))
        op_income = _safe_float(item.get("operatingIncome"))
        net = _safe_float(item.get("netIncome"))
        ebitda = _safe_float(item.get("ebitda"))
        eps = _safe_float(item.get("eps"))
        eps_diluted = _safe_float(item.get("epsDiluted")) or _safe_float(item.get("epsdiluted"))
        shares = _safe_float(item.get("weightedAverageShsOutDil"))

        gross_margin = round(gross / revenue * 100, 2) if gross and revenue and revenue != 0 else None
        op_margin = round(op_income / revenue * 100, 2) if op_income and revenue and revenue != 0 else None
        net_margin = round(net / revenue * 100, 2) if net and revenue and revenue != 0 else None

        result.append({
            "date": item.get("date", ""),
            "revenue": revenue,
            "revenue_formatted": _format_large_number(revenue),
            "gross_profit": gross,
            "operating_income": op_income,
            "net_income": net,
            "net_income_formatted": _format_large_number(net),
            "eps": eps_diluted or eps,
            "eps_diluted": eps_diluted,
            "gross_margin": gross_margin,
            "operating_margin": op_margin,
            "net_margin": net_margin,
            "ebitda": ebitda,
            "weighted_avg_shares": shares,
            "data_source": "fmp",
        })
    return result


def get_income_statements(ticker: str, period: str = "quarter", limit: int = 8):
    ck = f"income_{ticker}_{period}_{limit}"
    cached = _cached(ck, 3600)
    if cached:
        return cached

    result = []

    fmp_params = {"symbol": ticker, "limit": str(limit)}
    if period == "quarter":
        fmp_params["period"] = "quarter"
    fmp_data = _fmp_get("income-statement", fmp_params)
    if fmp_data and isinstance(fmp_data, list) and len(fmp_data) > 0:
        result = _parse_fmp_income(fmp_data, limit)

    if not result:
        yf_ticker = _yf_get_ticker(ticker)
        if yf_ticker:
            try:
                if period == "quarter":
                    df = yf_ticker.quarterly_income_stmt
                else:
                    df = yf_ticker.income_stmt
                if df is not None and not df.empty:
                    for col in list(df.columns)[:limit]:
                        date_str = col.strftime("%Y-%m-%d") if hasattr(col, 'strftime') else str(col)[:10]
                        revenue = _safe_float(df.loc["Total Revenue", col]) if "Total Revenue" in df.index else None
                        gross = _safe_float(df.loc["Gross Profit", col]) if "Gross Profit" in df.index else None
                        op_income = _safe_float(df.loc["Operating Income", col]) if "Operating Income" in df.index else None
                        net = _safe_float(df.loc["Net Income", col]) if "Net Income" in df.index else None
                        ebitda = _safe_float(df.loc["EBITDA", col]) if "EBITDA" in df.index else None
                        eps = _safe_float(df.loc["Diluted EPS", col]) if "Diluted EPS" in df.index else (
                            _safe_float(df.loc["Basic EPS", col]) if "Basic EPS" in df.index else None
                        )
                        shares = _safe_float(df.loc["Diluted Average Shares", col]) if "Diluted Average Shares" in df.index else None

                        gross_margin = round(gross / revenue * 100, 2) if gross and revenue and revenue != 0 else None
                        op_margin = round(op_income / revenue * 100, 2) if op_income and revenue and revenue != 0 else None
                        net_margin = round(net / revenue * 100, 2) if net and revenue and revenue != 0 else None

                        result.append({
                            "date": date_str,
                            "revenue": revenue,
                            "revenue_formatted": _format_large_number(revenue),
                            "gross_profit": gross,
                            "operating_income": op_income,
                            "net_income": net,
                            "net_income_formatted": _format_large_number(net),
                            "eps": eps,
                            "eps_diluted": eps,
                            "gross_margin": gross_margin,
                            "operating_margin": op_margin,
                            "net_margin": net_margin,
                            "ebitda": ebitda,
                            "weighted_avg_shares": shares,
                            "data_source": "yfinance",
                        })
            except Exception as e:
                logger.warning(f"yfinance income statement error for {ticker}: {e}")

    if result:
        _set_cache(ck, result)
    return result


def get_balance_sheet(ticker: str, period: str = "quarter", limit: int = 4):
    ck = f"balance_{ticker}_{period}_{limit}"
    cached = _cached(ck, 3600)
    if cached:
        return cached

    result = []

    fmp_params = {"symbol": ticker, "limit": str(limit)}
    if period == "quarter":
        fmp_params["period"] = "quarter"
    fmp_data = _fmp_get("balance-sheet-statement", fmp_params)
    if fmp_data and isinstance(fmp_data, list) and len(fmp_data) > 0:
        for item in fmp_data[:limit]:
            total_assets = _safe_float(item.get("totalAssets"))
            total_liabilities = _safe_float(item.get("totalLiabilities"))
            total_equity = _safe_float(item.get("totalStockholdersEquity"))
            cash = _safe_float(item.get("cashAndCashEquivalents")) or _safe_float(item.get("cashAndShortTermInvestments"))
            total_debt = _safe_float(item.get("totalDebt")) or _safe_float(item.get("longTermDebt"))
            current_assets = _safe_float(item.get("totalCurrentAssets"))
            current_liabilities = _safe_float(item.get("totalCurrentLiabilities"))
            retained = _safe_float(item.get("retainedEarnings"))
            goodwill = _safe_float(item.get("goodwill"))
            net_debt = _safe_float(item.get("netDebt"))
            if net_debt is None and total_debt is not None and cash is not None:
                net_debt = total_debt - cash
            result.append({
                "date": item.get("date", ""),
                "total_assets": total_assets,
                "total_assets_formatted": _format_large_number(total_assets),
                "total_liabilities": total_liabilities,
                "total_liabilities_formatted": _format_large_number(total_liabilities),
                "total_equity": total_equity,
                "total_equity_formatted": _format_large_number(total_equity),
                "cash_and_equivalents": cash,
                "total_debt": total_debt,
                "net_debt": net_debt,
                "current_assets": current_assets,
                "current_liabilities": current_liabilities,
                "retained_earnings": retained,
                "goodwill": goodwill,
                "data_source": "fmp",
            })

    if not result:
        yf_ticker = _yf_get_ticker(ticker)
        if yf_ticker:
            try:
                if period == "quarter":
                    df = yf_ticker.quarterly_balance_sheet
                else:
                    df = yf_ticker.balance_sheet
                if df is not None and not df.empty:
                    def _get(row):
                        return _safe_float(df.loc[row, col]) if row in df.index else None

                    for col in list(df.columns)[:limit]:
                        date_str = col.strftime("%Y-%m-%d") if hasattr(col, 'strftime') else str(col)[:10]
                        total_assets = _get("Total Assets")
                        total_liabilities = _get("Total Liabilities Net Minority Interest")
                        if total_liabilities is None:
                            total_liabilities = _get("Total Non Current Liabilities Net Minority Interest")
                        total_equity = _get("Stockholders Equity") or _get("Total Equity Gross Minority Interest")
                        cash = _get("Cash And Cash Equivalents") or _get("Cash Cash Equivalents And Short Term Investments")
                        total_debt = _get("Total Debt") or _get("Long Term Debt")
                        current_assets = _get("Current Assets")
                        current_liabilities = _get("Current Liabilities")
                        retained = _get("Retained Earnings")
                        goodwill = _get("Goodwill")
                        net_debt = (total_debt - cash) if total_debt is not None and cash is not None else None

                        result.append({
                            "date": date_str,
                            "total_assets": total_assets,
                            "total_assets_formatted": _format_large_number(total_assets),
                            "total_liabilities": total_liabilities,
                            "total_liabilities_formatted": _format_large_number(total_liabilities),
                            "total_equity": total_equity,
                            "total_equity_formatted": _format_large_number(total_equity),
                            "cash_and_equivalents": cash,
                            "total_debt": total_debt,
                            "net_debt": net_debt,
                            "current_assets": current_assets,
                            "current_liabilities": current_liabilities,
                            "retained_earnings": retained,
                            "goodwill": goodwill,
                        })
            except Exception as e:
                logger.warning(f"yfinance balance sheet error for {ticker}: {e}")

    if result:
        _set_cache(ck, result)
    return result


def get_cash_flow(ticker: str, period: str = "quarter", limit: int = 4):
    ck = f"cashflow_{ticker}_{period}_{limit}"
    cached = _cached(ck, 3600)
    if cached:
        return cached

    result = []

    fmp_params = {"symbol": ticker, "limit": str(limit)}
    if period == "quarter":
        fmp_params["period"] = "quarter"
    fmp_data = _fmp_get("cash-flow-statement", fmp_params)
    if fmp_data and isinstance(fmp_data, list) and len(fmp_data) > 0:
        for item in fmp_data[:limit]:
            operating = _safe_float(item.get("netCashProvidedByOperatingActivities")) or _safe_float(item.get("operatingCashFlow"))
            investing = _safe_float(item.get("netCashProvidedByInvestingActivities")) or _safe_float(item.get("netCashUsedForInvestingActivites"))
            financing = _safe_float(item.get("netCashProvidedByFinancingActivities")) or _safe_float(item.get("netCashUsedProvidedByFinancingActivities"))
            capex = _safe_float(item.get("capitalExpenditure"))
            fcf = _safe_float(item.get("freeCashFlow"))
            if fcf is None and operating is not None and capex is not None:
                fcf = operating - abs(capex)
            dividends = _safe_float(item.get("dividendsPaid"))
            buyback = _safe_float(item.get("commonStockRepurchased"))

            result.append({
                "date": item.get("date", ""),
                "operating_cash_flow": operating,
                "operating_formatted": _format_large_number(operating),
                "investing_cash_flow": investing,
                "investing_formatted": _format_large_number(investing),
                "financing_cash_flow": financing,
                "financing_formatted": _format_large_number(financing),
                "capital_expenditure": capex,
                "free_cash_flow": fcf,
                "fcf_formatted": _format_large_number(fcf),
                "dividends_paid": dividends,
                "stock_repurchased": buyback,
                "data_source": "fmp",
            })

    if not result:
        yf_ticker = _yf_get_ticker(ticker)
        if yf_ticker:
            try:
                if period == "quarter":
                    df = yf_ticker.quarterly_cashflow
                else:
                    df = yf_ticker.cashflow
                if df is not None and not df.empty:
                    def _get(row):
                        return _safe_float(df.loc[row, col]) if row in df.index else None

                    for col in list(df.columns)[:limit]:
                        date_str = col.strftime("%Y-%m-%d") if hasattr(col, 'strftime') else str(col)[:10]
                        operating = _get("Operating Cash Flow") or _get("Cash Flow From Continuing Operating Activities")
                        investing = _get("Investing Cash Flow") or _get("Cash Flow From Continuing Investing Activities")
                        financing = _get("Financing Cash Flow") or _get("Cash Flow From Continuing Financing Activities")
                        capex = _get("Capital Expenditure")
                        fcf = _get("Free Cash Flow")
                        if fcf is None and operating is not None and capex is not None:
                            fcf = operating + capex
                        dividends = _get("Cash Dividends Paid") or _get("Common Stock Dividend Paid")
                        buyback = _get("Repurchase Of Capital Stock") or _get("Common Stock Payments")

                        result.append({
                            "date": date_str,
                            "operating_cash_flow": operating,
                            "operating_formatted": _format_large_number(operating),
                            "investing_cash_flow": investing,
                            "investing_formatted": _format_large_number(investing),
                            "financing_cash_flow": financing,
                            "financing_formatted": _format_large_number(financing),
                            "capital_expenditure": capex,
                            "free_cash_flow": fcf,
                            "fcf_formatted": _format_large_number(fcf),
                            "dividends_paid": dividends,
                            "stock_repurchased": buyback,
                        })
            except Exception as e:
                logger.warning(f"yfinance cash flow error for {ticker}: {e}")

    if result:
        _set_cache(ck, result)
    return result


def get_key_ratios(ticker: str):
    ck = f"ratios_{ticker}"
    cached = _cached(ck, 3600)
    if cached:
        return cached

    result = {}
    metrics = _finnhub_get("/stock/metric", {"symbol": ticker, "metric": "all"})
    if metrics and isinstance(metrics, dict):
        m = metrics.get("metric", {})
        result = {
            "pe_ratio": _safe_float(m.get("peBasicExclExtraTTM")) or _safe_float(m.get("peTTM")),
            "pb_ratio": _safe_float(m.get("pbQuarterly")) or _safe_float(m.get("pbAnnual")),
            "ps_ratio": _safe_float(m.get("psTTM")) or _safe_float(m.get("psAnnual")),
            "peg_ratio": _safe_float(m.get("pegRatio")),
            "roe": _safe_float(m.get("roeTTM")) or _safe_float(m.get("roeRfy")),
            "roa": _safe_float(m.get("roaTTM")) or _safe_float(m.get("roaRfy")),
            "debt_to_equity": _safe_float(m.get("totalDebt/totalEquityQuarterly")) or _safe_float(m.get("totalDebt/totalEquityAnnual")),
            "current_ratio": _safe_float(m.get("currentRatioQuarterly")) or _safe_float(m.get("currentRatioAnnual")),
            "quick_ratio": _safe_float(m.get("quickRatioQuarterly")) or _safe_float(m.get("quickRatioAnnual")),
            "dividend_yield": _safe_float(m.get("dividendYieldIndicatedAnnual")),
            "payout_ratio": _safe_float(m.get("payoutRatioTTM")) or _safe_float(m.get("payoutRatioAnnual")),
            "gross_margin": _safe_float(m.get("grossMarginTTM")) or _safe_float(m.get("grossMargin5Y")),
            "operating_margin": _safe_float(m.get("operatingMarginTTM")) or _safe_float(m.get("operatingMargin5Y")),
            "net_margin": _safe_float(m.get("netProfitMarginTTM")) or _safe_float(m.get("netProfitMargin5Y")),
            "asset_turnover": _safe_float(m.get("assetTurnoverTTM")) or _safe_float(m.get("assetTurnoverAnnual")),
            "inventory_turnover": _safe_float(m.get("inventoryTurnoverTTM")) or _safe_float(m.get("inventoryTurnoverAnnual")),
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        }

    if not result:
        yf_ticker = _yf_get_ticker(ticker)
        if yf_ticker:
            try:
                info = yf_ticker.info or {}
                result = {
                    "pe_ratio": _safe_float(info.get("trailingPE")),
                    "pb_ratio": _safe_float(info.get("priceToBook")),
                    "ps_ratio": _safe_float(info.get("priceToSalesTrailing12Months")),
                    "peg_ratio": _safe_float(info.get("pegRatio")),
                    "roe": round(_safe_float(info.get("returnOnEquity"), 0) * 100, 2) if info.get("returnOnEquity") else None,
                    "roa": round(_safe_float(info.get("returnOnAssets"), 0) * 100, 2) if info.get("returnOnAssets") else None,
                    "debt_to_equity": _safe_float(info.get("debtToEquity")),
                    "current_ratio": _safe_float(info.get("currentRatio")),
                    "quick_ratio": _safe_float(info.get("quickRatio")),
                    "dividend_yield": round(_safe_float(info.get("dividendYield"), 0) * 100, 2) if info.get("dividendYield") else None,
                    "payout_ratio": round(_safe_float(info.get("payoutRatio"), 0) * 100, 2) if info.get("payoutRatio") else None,
                    "gross_margin": round(_safe_float(info.get("grossMargins"), 0) * 100, 2) if info.get("grossMargins") else None,
                    "operating_margin": round(_safe_float(info.get("operatingMargins"), 0) * 100, 2) if info.get("operatingMargins") else None,
                    "net_margin": round(_safe_float(info.get("profitMargins"), 0) * 100, 2) if info.get("profitMargins") else None,
                    "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                }
            except Exception as e:
                logger.warning(f"yfinance ratios error for {ticker}: {e}")

    if result:
        _set_cache(ck, result)
    return result


def get_key_metrics(ticker: str):
    ck = f"metrics_{ticker}"
    cached = _cached(ck, 3600)
    if cached:
        return cached

    result = {}

    fmp_ttm = _fmp_get("key-metrics-ttm", {"symbol": ticker})
    if fmp_ttm and isinstance(fmp_ttm, list) and len(fmp_ttm) > 0:
        t = fmp_ttm[0]
        mkt_cap = _safe_float(t.get("marketCap")) or _safe_float(t.get("marketCapTTM"))
        ev = _safe_float(t.get("enterpriseValueTTM"))
        ebitda = _safe_float(t.get("evToEBITDATTM")) or _safe_float(t.get("enterpriseValueOverEBITDATTM"))
        fcf_yield = _safe_float(t.get("freeCashFlowYieldTTM"))
        if fcf_yield and fcf_yield < 1:
            fcf_yield = round(fcf_yield * 100, 2)
        result = {
            "market_cap": mkt_cap,
            "market_cap_formatted": _format_large_number(mkt_cap),
            "enterprise_value": ev,
            "ev_formatted": _format_large_number(ev),
            "ev_to_ebitda": ebitda,
            "ev_to_revenue": _safe_float(t.get("evToSalesTTM")),
            "fcf_yield": fcf_yield,
            "earnings_yield": _safe_float(t.get("earningsYieldTTM")),
            "book_value_per_share": _safe_float(t.get("bookValuePerShareTTM")),
            "tangible_book_per_share": _safe_float(t.get("tangibleBookValuePerShareTTM")),
            "revenue_per_share": _safe_float(t.get("revenuePerShareTTM")),
            "pe_ratio": _safe_float(t.get("peRatioTTM")),
            "pb_ratio": _safe_float(t.get("pbRatioTTM")),
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "data_source": "fmp",
        }
        finnhub_metrics = _finnhub_get("/stock/metric", {"symbol": ticker, "metric": "all"})
        if finnhub_metrics and isinstance(finnhub_metrics, dict):
            m = finnhub_metrics.get("metric", {})
            if not result.get("pe_ratio"):
                result["pe_ratio"] = _safe_float(m.get("peTTM")) or _safe_float(m.get("peBasicExclExtraTTM"))
            if not result.get("pb_ratio"):
                result["pb_ratio"] = _safe_float(m.get("pbQuarterly")) or _safe_float(m.get("pbAnnual"))
            if not result.get("52_week_high"):
                result["52_week_high"] = _safe_float(m.get("52WeekHigh"))
            if not result.get("52_week_low"):
                result["52_week_low"] = _safe_float(m.get("52WeekLow"))
            if not result.get("beta"):
                result["beta"] = _safe_float(m.get("beta"))

    if not result:
        metrics = _finnhub_get("/stock/metric", {"symbol": ticker, "metric": "all"})
        if metrics and isinstance(metrics, dict):
            m = metrics.get("metric", {})
            mkt_cap = _safe_float(m.get("marketCapitalization"))
            if mkt_cap:
                mkt_cap = mkt_cap * 1e6
            result = {
                "market_cap": mkt_cap,
                "market_cap_formatted": _format_large_number(mkt_cap),
                "enterprise_value": None,
                "ev_formatted": None,
                "ev_to_ebitda": _safe_float(m.get("enterpriseValueEBITDATTM")),
                "ev_to_revenue": None,
                "fcf_yield": _safe_float(m.get("freeCashFlowYieldTTM")),
                "earnings_yield": None,
                "book_value_per_share": _safe_float(m.get("bookValuePerShareQuarterly")) or _safe_float(m.get("bookValuePerShareAnnual")),
                "tangible_book_per_share": _safe_float(m.get("tangibleBookValuePerShareQuarterly")),
                "revenue_per_share": _safe_float(m.get("revenuePerShareTTM")) or _safe_float(m.get("revenuePerShareAnnual")),
                "pe_ratio": _safe_float(m.get("peTTM")),
                "pb_ratio": _safe_float(m.get("pbQuarterly")),
                "52_week_high": _safe_float(m.get("52WeekHigh")),
                "52_week_low": _safe_float(m.get("52WeekLow")),
                "beta": _safe_float(m.get("beta")),
                "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            }

    if not result:
        yf_ticker = _yf_get_ticker(ticker)
        if yf_ticker:
            try:
                info = yf_ticker.info or {}
                mkt_cap = _safe_float(info.get("marketCap"))
                ev = _safe_float(info.get("enterpriseValue"))
                result = {
                    "market_cap": mkt_cap,
                    "market_cap_formatted": _format_large_number(mkt_cap),
                    "enterprise_value": ev,
                    "ev_formatted": _format_large_number(ev),
                    "ev_to_ebitda": _safe_float(info.get("enterpriseToEbitda")),
                    "ev_to_revenue": _safe_float(info.get("enterpriseToRevenue")),
                    "fcf_yield": round(_safe_float(info.get("freeCashflow"), 0) / max(mkt_cap, 1) * 100, 2) if mkt_cap and info.get("freeCashflow") else None,
                    "earnings_yield": round(1 / max(_safe_float(info.get("trailingPE"), 999), 0.01) * 100, 2) if info.get("trailingPE") else None,
                    "book_value_per_share": _safe_float(info.get("bookValue")),
                    "revenue_per_share": _safe_float(info.get("revenuePerShare")),
                    "pe_ratio": _safe_float(info.get("trailingPE")),
                    "pb_ratio": _safe_float(info.get("priceToBook")),
                    "52_week_high": _safe_float(info.get("fiftyTwoWeekHigh")),
                    "52_week_low": _safe_float(info.get("fiftyTwoWeekLow")),
                    "beta": _safe_float(info.get("beta")),
                    "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                }
            except Exception as e:
                logger.warning(f"yfinance metrics error for {ticker}: {e}")

    if result:
        _set_cache(ck, result)
    return result


def get_analyst_estimates(ticker: str, limit: int = 4):
    ck = f"estimates_{ticker}_{limit}"
    cached = _cached(ck, 1800)
    if cached:
        return cached

    result = []
    earnings_data = _finnhub_get("/stock/earnings", {"symbol": ticker})
    if earnings_data and isinstance(earnings_data, list):
        for item in earnings_data[:limit]:
            result.append({
                "date": item.get("period"),
                "estimated_eps_avg": _safe_float(item.get("estimate")),
                "actual_eps": _safe_float(item.get("actual")),
                "eps_surprise": _safe_float(item.get("surprise")),
                "eps_surprise_pct": _safe_float(item.get("surprisePercent")),
                "quarter": item.get("quarter"),
            })

    if not result:
        yf_ticker = _yf_get_ticker(ticker)
        if yf_ticker:
            try:
                info = yf_ticker.info or {}
                if info.get("targetMeanPrice") or info.get("recommendationMean"):
                    result.append({
                        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                        "estimated_eps_avg": _safe_float(info.get("forwardEps")),
                        "target_price": _safe_float(info.get("targetMeanPrice")),
                        "target_high": _safe_float(info.get("targetHighPrice")),
                        "target_low": _safe_float(info.get("targetLowPrice")),
                        "recommendation": info.get("recommendationKey"),
                        "num_analysts": _safe_float(info.get("numberOfAnalystOpinions")),
                    })
            except Exception:
                pass

    _set_cache(ck, result)
    return result


def get_company_overview(ticker: str):
    ck = f"overview_{ticker}"
    cached = _cached(ck, 3600)
    if cached:
        return cached

    fmp_profile = _fmp_get("profile", {"symbol": ticker})
    if fmp_profile and isinstance(fmp_profile, list) and len(fmp_profile) > 0:
        p = fmp_profile[0]
        mkt_cap = _safe_float(p.get("marketCap")) or _safe_float(p.get("mktCap"))
        price = _safe_float(p.get("price"))
        eps = _safe_float(p.get("eps"))
        pe_ratio = None
        if price and eps and eps > 0:
            pe_ratio = round(price / eps, 2)

        fmp_ttm = _fmp_get("key-metrics-ttm", {"symbol": ticker})
        if fmp_ttm and isinstance(fmp_ttm, list) and len(fmp_ttm) > 0:
            ttm = fmp_ttm[0]
            pe_from_ttm = _safe_float(ttm.get("peRatioTTM"))
            if pe_from_ttm:
                pe_ratio = round(pe_from_ttm, 2)
            if not eps:
                eps_from_ttm = _safe_float(ttm.get("netIncomePerShareTTM"))
                if eps_from_ttm:
                    eps = round(eps_from_ttm, 2)
            earnings_yield = _safe_float(ttm.get("earningsYieldTTM"))
            if not pe_ratio and earnings_yield and earnings_yield != 0:
                pe_ratio = round(1.0 / earnings_yield, 2)
            if not eps and pe_ratio and price and pe_ratio != 0:
                eps = round(price / pe_ratio, 2)

        if not pe_ratio or not eps:
            yf_ticker = _yf_get_ticker(ticker)
            if yf_ticker:
                try:
                    info = yf_ticker.info or {}
                    if not pe_ratio:
                        pe_ratio = round(_safe_float(info.get("trailingPE")) or 0, 2) or None
                    if not eps:
                        eps = round(_safe_float(info.get("trailingEps")) or 0, 2) or None
                except Exception:
                    pass

        result = {
            "name": p.get("companyName"),
            "symbol": p.get("symbol"),
            "exchange": p.get("exchangeShortName") or p.get("exchange"),
            "sector": p.get("sector"),
            "industry": p.get("industry"),
            "description": (p.get("description") or "")[:500],
            "ceo": p.get("ceo"),
            "employees": _safe_float(p.get("fullTimeEmployees")),
            "country": p.get("country"),
            "website": p.get("website"),
            "ipo_date": p.get("ipoDate"),
            "market_cap": mkt_cap,
            "market_cap_formatted": _format_large_number(mkt_cap),
            "pe_ratio": pe_ratio,
            "eps": eps,
            "beta": _safe_float(p.get("beta")),
            "price": price,
            "52w_high": _safe_float(p.get("range", "").split("-")[-1].strip()) if p.get("range") and "-" in str(p.get("range", "")) else None,
            "52w_low": _safe_float(p.get("range", "").split("-")[0].strip()) if p.get("range") and "-" in str(p.get("range", "")) else None,
            "avg_volume": _safe_float(p.get("volAvg")),
            "dividend_yield": round(_safe_float(p.get("lastDiv"), 0) / max(_safe_float(p.get("price"), 1), 0.01) * 100, 2) if _safe_float(p.get("lastDiv")) and _safe_float(p.get("price")) else None,
            "data_source": "fmp",
        }
        _set_cache(ck, result)
        return result

    profile = _finnhub_get("/stock/profile2", {"symbol": ticker})
    if profile and isinstance(profile, dict) and profile.get("name"):
        yf_ticker = _yf_get_ticker(ticker)
        info = {}
        if yf_ticker:
            try:
                info = yf_ticker.info or {}
            except Exception:
                pass
        mkt_cap = _safe_float(profile.get("marketCapitalization"))
        if mkt_cap:
            mkt_cap = mkt_cap * 1e6
        pe_ratio = _safe_float(info.get("trailingPE")) or _safe_float(info.get("forwardPE"))
        eps = _safe_float(info.get("trailingEps"))
        result = {
            "name": profile.get("name"),
            "symbol": profile.get("ticker"),
            "exchange": profile.get("exchange"),
            "sector": info.get("sector") or profile.get("finnhubIndustry"),
            "industry": info.get("industry") or profile.get("finnhubIndustry"),
            "description": (info.get("longBusinessSummary") or "")[:500],
            "ceo": info.get("companyOfficers", [{}])[0].get("name") if info.get("companyOfficers") else None,
            "employees": info.get("fullTimeEmployees"),
            "country": profile.get("country"),
            "website": profile.get("weburl"),
            "ipo_date": profile.get("ipo"),
            "market_cap": mkt_cap,
            "market_cap_formatted": _format_large_number(mkt_cap),
            "pe_ratio": round(pe_ratio, 2) if pe_ratio else None,
            "eps": round(eps, 2) if eps else None,
            "beta": _safe_float(info.get("beta")),
            "price": _safe_float(info.get("currentPrice")) or _safe_float(info.get("previousClose")),
            "52w_high": _safe_float(info.get("fiftyTwoWeekHigh")),
            "52w_low": _safe_float(info.get("fiftyTwoWeekLow")),
            "avg_volume": _safe_float(info.get("averageVolume")),
            "dividend_yield": round(_safe_float(info.get("dividendYield"), 0) * 100, 2) if info.get("dividendYield") else None,
        }
        _set_cache(ck, result)
        return result

    av_data = _av_get("OVERVIEW", {"symbol": ticker})
    if av_data:
        result = {
            "name": av_data.get("Name"),
            "symbol": av_data.get("Symbol"),
            "exchange": av_data.get("Exchange"),
            "sector": av_data.get("Sector"),
            "industry": av_data.get("Industry"),
            "description": (av_data.get("Description") or "")[:500],
            "employees": av_data.get("FullTimeEmployees"),
            "country": av_data.get("Country"),
            "market_cap": _safe_float(av_data.get("MarketCapitalization")),
            "market_cap_formatted": _format_large_number(_safe_float(av_data.get("MarketCapitalization"))),
            "beta": _safe_float(av_data.get("Beta")),
            "pe_ratio": _safe_float(av_data.get("PERatio")),
            "peg_ratio": _safe_float(av_data.get("PEGRatio")),
            "dividend_yield": round(_safe_float(av_data.get("DividendYield"), 0) * 100, 2),
            "eps": _safe_float(av_data.get("EPS")),
            "52w_high": _safe_float(av_data.get("52WeekHigh")),
            "52w_low": _safe_float(av_data.get("52WeekLow")),
        }
        _set_cache(ck, result)
        return result

    return {}


def _compute_piotroski(income, balance, cashflow) -> Dict[str, Any]:
    if not income or not balance or not cashflow or len(income) < 2 or len(balance) < 2 or len(cashflow) < 1:
        return {"score": None, "components": {}, "interpretation": "Insufficient data"}

    curr_inc = income[0]
    prev_inc = income[1]
    curr_bal = balance[0]
    prev_bal = balance[1]
    curr_cf = cashflow[0]

    components = {}
    score = 0

    ni = _safe_float(curr_inc.get("net_income"), 0)
    components["positive_net_income"] = ni > 0
    if ni > 0:
        score += 1

    ocf = _safe_float(curr_cf.get("operating_cash_flow"), 0)
    components["positive_operating_cf"] = ocf > 0
    if ocf > 0:
        score += 1

    prev_roa = _safe_float(prev_inc.get("net_income"), 0) / max(abs(_safe_float(prev_bal.get("total_assets"), 1)), 1)
    curr_roa = ni / max(abs(_safe_float(curr_bal.get("total_assets"), 1)), 1)
    components["increasing_roa"] = curr_roa > prev_roa
    if curr_roa > prev_roa:
        score += 1

    components["cf_exceeds_ni"] = ocf > ni
    if ocf > ni:
        score += 1

    curr_debt_ratio = _safe_float(curr_bal.get("total_debt"), 0) / max(abs(_safe_float(curr_bal.get("total_assets"), 1)), 1)
    prev_debt_ratio = _safe_float(prev_bal.get("total_debt"), 0) / max(abs(_safe_float(prev_bal.get("total_assets"), 1)), 1)
    components["decreasing_leverage"] = curr_debt_ratio <= prev_debt_ratio
    if curr_debt_ratio <= prev_debt_ratio:
        score += 1

    curr_cr = _safe_float(curr_bal.get("current_assets"), 0) / max(abs(_safe_float(curr_bal.get("current_liabilities"), 1)), 1)
    prev_cr = _safe_float(prev_bal.get("current_assets"), 0) / max(abs(_safe_float(prev_bal.get("current_liabilities"), 1)), 1)
    components["increasing_liquidity"] = curr_cr >= prev_cr
    if curr_cr >= prev_cr:
        score += 1

    curr_shares = _safe_float(curr_inc.get("weighted_avg_shares"), 0)
    prev_shares = _safe_float(prev_inc.get("weighted_avg_shares"), 0)
    components["no_dilution"] = curr_shares <= prev_shares if prev_shares > 0 else True
    if curr_shares <= prev_shares or prev_shares == 0:
        score += 1

    curr_gm = _safe_float(curr_inc.get("gross_margin"), 0)
    prev_gm = _safe_float(prev_inc.get("gross_margin"), 0)
    components["increasing_gross_margin"] = curr_gm >= prev_gm
    if curr_gm >= prev_gm:
        score += 1

    curr_at = _safe_float(curr_inc.get("revenue"), 0) / max(abs(_safe_float(curr_bal.get("total_assets"), 1)), 1)
    prev_at = _safe_float(prev_inc.get("revenue"), 0) / max(abs(_safe_float(prev_bal.get("total_assets"), 1)), 1)
    components["increasing_asset_turnover"] = curr_at >= prev_at
    if curr_at >= prev_at:
        score += 1

    if score >= 7:
        interp = "Strong"
    elif score >= 5:
        interp = "Moderate"
    elif score >= 3:
        interp = "Weak"
    else:
        interp = "Very Weak"

    return {"score": score, "max_score": 9, "components": components, "interpretation": interp}


def _compute_altman_z(balance, income) -> Dict[str, Any]:
    if not balance or not income or len(balance) < 1 or len(income) < 1:
        return {"score": None, "interpretation": "Insufficient data", "components": {}}

    bal = balance[0]
    inc = income[0]

    ta = max(abs(_safe_float(bal.get("total_assets"), 1)), 1)
    ca = _safe_float(bal.get("current_assets"), 0)
    cl = _safe_float(bal.get("current_liabilities"), 0)
    re = _safe_float(bal.get("retained_earnings"), 0)
    ebit = _safe_float(inc.get("operating_income"), 0)
    mkt_cap = _safe_float(bal.get("total_equity"), 0)
    tl = _safe_float(bal.get("total_liabilities"), 0)
    revenue = _safe_float(inc.get("revenue"), 0)

    wc_ta = (ca - cl) / ta
    re_ta = re / ta
    ebit_ta = ebit / ta
    eq_tl = mkt_cap / max(abs(tl), 1)
    rev_ta = revenue / ta

    z = 1.2 * wc_ta + 1.4 * re_ta + 3.3 * ebit_ta + 0.6 * eq_tl + 1.0 * rev_ta

    if z > 2.99:
        interp = "Safe Zone"
    elif z > 1.81:
        interp = "Grey Zone"
    else:
        interp = "Distress Zone"

    return {
        "score": round(z, 2),
        "interpretation": interp,
        "components": {
            "working_capital_to_assets": round(wc_ta, 4),
            "retained_earnings_to_assets": round(re_ta, 4),
            "ebit_to_assets": round(ebit_ta, 4),
            "equity_to_liabilities": round(eq_tl, 4),
            "revenue_to_assets": round(rev_ta, 4),
        },
    }


def _compute_growth(income_statements):
    if not income_statements or len(income_statements) < 2:
        return {}

    growth = []
    for i in range(len(income_statements) - 1):
        curr = income_statements[i]
        prev = income_statements[i + 1]

        rev_curr = _safe_float(curr.get("revenue"), 0)
        rev_prev = _safe_float(prev.get("revenue"), 0)
        ni_curr = _safe_float(curr.get("net_income"), 0)
        ni_prev = _safe_float(prev.get("net_income"), 0)
        eps_curr = _safe_float(curr.get("eps_diluted"), 0)
        eps_prev = _safe_float(prev.get("eps_diluted"), 0)

        rev_growth = round((rev_curr - rev_prev) / max(abs(rev_prev), 1) * 100, 2) if rev_prev != 0 else None
        ni_growth = round((ni_curr - ni_prev) / max(abs(ni_prev), 1) * 100, 2) if ni_prev != 0 else None
        eps_growth = round((eps_curr - eps_prev) / max(abs(eps_prev), 0.01) * 100, 2) if eps_prev != 0 else None

        growth.append({
            "date": curr.get("date"),
            "revenue_growth_pct": rev_growth,
            "net_income_growth_pct": ni_growth,
            "eps_growth_pct": eps_growth,
        })

    return growth


def get_fundamental_analysis(ticker: str) -> Dict[str, Any]:
    ck = f"fundamental_full_{ticker}"
    cached = _cached(ck, 600)
    if cached:
        return cached

    ticker = ticker.upper()

    overview = get_company_overview(ticker)
    income = get_income_statements(ticker, "quarter", 8)
    balance = get_balance_sheet(ticker, "quarter", 4)
    cashflow = get_cash_flow(ticker, "quarter", 4)
    ratios = get_key_ratios(ticker)
    metrics = get_key_metrics(ticker)
    estimates = get_analyst_estimates(ticker, 4)

    growth = _compute_growth(income)

    piotroski = _compute_piotroski(income, balance, cashflow)
    altman_z = _compute_altman_z(balance, income)

    revenue_trend = []
    for stmt in reversed(income):
        if stmt.get("date") and stmt.get("revenue") is not None:
            revenue_trend.append({
                "date": stmt["date"],
                "revenue": stmt["revenue"],
                "revenue_formatted": stmt.get("revenue_formatted"),
                "net_income": stmt.get("net_income"),
                "net_income_formatted": stmt.get("net_income_formatted"),
                "eps": stmt.get("eps_diluted"),
                "gross_margin": stmt.get("gross_margin"),
                "operating_margin": stmt.get("operating_margin"),
                "net_margin": stmt.get("net_margin"),
            })

    cf_waterfall = []
    if cashflow and len(cashflow) > 0:
        latest_cf = cashflow[0]
        cf_waterfall = [
            {"label": "Operating", "value": latest_cf.get("operating_cash_flow"), "formatted": latest_cf.get("operating_formatted"), "type": "positive" if (_safe_float(latest_cf.get("operating_cash_flow"), 0)) >= 0 else "negative"},
            {"label": "CapEx", "value": latest_cf.get("capital_expenditure"), "formatted": _format_large_number(latest_cf.get("capital_expenditure")), "type": "negative"},
            {"label": "Free Cash Flow", "value": latest_cf.get("free_cash_flow"), "formatted": latest_cf.get("fcf_formatted"), "type": "total"},
            {"label": "Investing", "value": latest_cf.get("investing_cash_flow"), "formatted": latest_cf.get("investing_formatted"), "type": "negative" if (_safe_float(latest_cf.get("investing_cash_flow"), 0)) < 0 else "positive"},
            {"label": "Financing", "value": latest_cf.get("financing_cash_flow"), "formatted": latest_cf.get("financing_formatted"), "type": "negative" if (_safe_float(latest_cf.get("financing_cash_flow"), 0)) < 0 else "positive"},
        ]

    estimates_vs_actual = []
    if estimates:
        for est in estimates:
            actual_eps = _safe_float(est.get("actual_eps"))
            est_eps = _safe_float(est.get("estimated_eps_avg"))
            eps_surprise = _safe_float(est.get("eps_surprise_pct"))
            if actual_eps is not None and est_eps is not None:
                if eps_surprise is None and est_eps != 0:
                    eps_surprise = round((actual_eps - est_eps) / abs(est_eps) * 100, 2)
                estimates_vs_actual.append({
                    "date": est.get("date"),
                    "actual_eps": actual_eps,
                    "estimated_eps": est_eps,
                    "eps_surprise_pct": eps_surprise,
                    "eps_beat": eps_surprise > 0 if eps_surprise is not None else None,
                })

    has_data = bool(overview or income or ratios or metrics)

    result = {
        "ticker": ticker,
        "has_data": has_data,
        "overview": overview,
        "ratios": ratios,
        "metrics": metrics,
        "revenue_trend": revenue_trend,
        "growth": growth,
        "cash_flow_waterfall": cf_waterfall,
        "estimates_vs_actual": estimates_vs_actual,
        "analyst_estimates": estimates,
        "financial_health": {
            "piotroski_f_score": piotroski,
            "altman_z_score": altman_z,
        },
        "balance_sheet_summary": {
            "total_assets": balance[0].get("total_assets") if balance else None,
            "total_assets_formatted": balance[0].get("total_assets_formatted") if balance else None,
            "total_liabilities": balance[0].get("total_liabilities") if balance else None,
            "total_liabilities_formatted": balance[0].get("total_liabilities_formatted") if balance else None,
            "total_equity": balance[0].get("total_equity") if balance else None,
            "total_equity_formatted": balance[0].get("total_equity_formatted") if balance else None,
            "cash": balance[0].get("cash_and_equivalents") if balance else None,
            "cash_formatted": _format_large_number(balance[0].get("cash_and_equivalents")) if balance else None,
            "total_debt": balance[0].get("total_debt") if balance else None,
            "total_debt_formatted": _format_large_number(balance[0].get("total_debt")) if balance else None,
        } if balance else {},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    _set_cache(ck, result)
    return result
