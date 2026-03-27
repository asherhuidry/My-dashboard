"""AI Deep Research endpoint — Claude with tool use for multi-source synthesis.

Claude acts as an autonomous research agent. It can:
  1. get_technical_analysis  — runs our feature pipeline
  2. get_news                — fetches yfinance + web news
  3. get_fundamentals        — earnings, insiders, options, key stats
  4. web_search              — general web search for any query
  5. get_macro_data          — Fear & Greed + market context

The endpoint runs a tool-use agentic loop (max 4 rounds) and returns a
structured research report.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from data.sources.news        import fetch_all_news, web_search, fetch_fear_greed
from data.sources.fundamentals import get_full_fundamental_profile, get_key_stats
from ml.patterns.features      import build_features

router = APIRouter()
log    = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 4


# ── Request / Response models ─────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    symbol:  str | None = None
    query:   str
    depth:   str = "standard"    # "quick" | "standard" | "deep"


class ResearchResponse(BaseModel):
    symbol:   str | None
    query:    str
    report:   str
    sources:  list[dict]
    data_used: list[str]
    model:    str
    tokens:   int


# ── Tool definitions for Claude ───────────────────────────────────────────────

TOOLS = [
    {
        "name":        "get_technical_analysis",
        "description": (
            "Fetch OHLCV data and compute 74 ML features for a stock/crypto symbol. "
            "Returns current price, key indicators (RSI, MACD, ADX, BB, EMA), signal, and returns."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Ticker (e.g. AAPL, BTC-USD)"},
                "days":   {"type": "integer", "description": "History days (default 180)", "default": 180},
            },
            "required": ["symbol"],
        },
    },
    {
        "name":        "get_news",
        "description":  (
            "Fetch recent news headlines and sentiment for a symbol from yfinance "
            "and web search. Returns articles with sentiment scores."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Ticker symbol"},
                "include_web": {"type": "boolean", "description": "Include broader web search", "default": True},
            },
            "required": ["symbol"],
        },
    },
    {
        "name":        "get_fundamentals",
        "description": (
            "Fetch fundamental data: earnings calendar, EPS history, insider trades, "
            "institutional holders, options summary, key financial ratios, and SEC filings."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Ticker symbol"},
            },
            "required": ["symbol"],
        },
    },
    {
        "name":        "web_search",
        "description": (
            "Search the web for any financial information, news, reports, or analysis. "
            "Use for general market questions, macro topics, or finding specific data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query":       {"type": "string",  "description": "Search query"},
                "max_results": {"type": "integer", "description": "Max results (default 5)", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name":        "get_macro_context",
        "description": (
            "Get current macro market context: Fear & Greed Index, "
            "and S&P 500 / VIX / DXY current snapshot."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
]


# ── Tool executor ─────────────────────────────────────────────────────────────

def _execute_tool(name: str, inputs: dict) -> tuple[Any, str]:
    """Run a tool and return (result_dict, source_label).

    Args:
        name:   Tool name matching TOOLS list.
        inputs: Dict of tool arguments.

    Returns:
        (result, source_label) tuple.
    """
    if name == "get_technical_analysis":
        sym  = inputs["symbol"].upper()
        days = int(inputs.get("days", 180))
        try:
            ticker = yf.Ticker(sym)
            period = "2y" if days > 365 else "1y"
            hist   = ticker.history(period=period)
            if hist.empty:
                return {"error": f"No data for {sym}"}, "yfinance"
            hist.index   = pd.to_datetime(hist.index, utc=True)
            hist.columns = [c.lower() for c in hist.columns]
            df           = hist[["open","high","low","close","volume"]].tail(days)
            feat_df      = build_features(df)
            row          = feat_df.iloc[-1]
            close        = df["close"]

            def sf(v):
                try:
                    f = float(v); return None if (np.isnan(f) or np.isinf(f)) else round(f, 4)
                except Exception: return None

            return {
                "symbol":    sym,
                "price":     sf(close.iloc[-1]),
                "return_1d": sf((close.iloc[-1]-close.iloc[-2])/close.iloc[-2]*100) if len(close)>1 else None,
                "return_5d": sf((close.iloc[-1]-close.iloc[-6])/close.iloc[-6]*100) if len(close)>5 else None,
                "rsi_14":    sf(row.get("rsi_14")),
                "macd":      sf(row.get("macd")),
                "macd_signal": sf(row.get("macd_signal")),
                "adx_14":    sf(row.get("adx_14")),
                "bb_pct":    sf(row.get("bb_pct")),
                "ema_9":     sf(row.get("ema_9")),
                "ema_21":    sf(row.get("ema_21")),
                "ema_50":    sf(row.get("ema_50")),
                "ema_200":   sf(row.get("ema_200")),
                "realized_vol_21d":  sf(row.get("realized_vol_21d")),
                "rolling_sharpe_21": sf(row.get("rolling_sharpe_21")),
                "max_drawdown_63":   sf(row.get("max_drawdown_63")),
                "stoch_k":   sf(row.get("stoch_k_14")),
                "williams_r":sf(row.get("williams_r_14")),
                "obv":       sf(row.get("obv")),
            }, "yfinance+features"
        except Exception as exc:
            return {"error": str(exc)}, "yfinance"

    elif name == "get_news":
        sym = inputs["symbol"].upper()
        include_web = bool(inputs.get("include_web", True))
        result = fetch_all_news(sym, include_web_search=include_web, max_yf=6, max_web=4)
        # Trim for token budget
        result["articles"] = result["articles"][:8]
        for a in result["articles"]:
            a.pop("summary", None)  # keep headlines only for token efficiency
        return result, "news+web"

    elif name == "get_fundamentals":
        sym = inputs["symbol"].upper()
        return get_full_fundamental_profile(sym), "sec+yfinance"

    elif name == "web_search":
        query  = inputs["query"]
        max_r  = int(inputs.get("max_results", 5))
        return {"results": web_search(query, max_results=max_r)}, "web_search"

    elif name == "get_macro_context":
        fg = fetch_fear_greed()
        macro = {"fear_greed": fg}
        # Quick snapshot of key market indices
        for sym, label in [("^GSPC","SP500"), ("^VIX","VIX"), ("DX-Y.NYB","DXY"), ("GC=F","Gold")]:
            try:
                t   = yf.Ticker(sym)
                inf = t.fast_info
                macro[label] = {
                    "price":    round(float(inf.get("last_price") or inf.get("regularMarketPrice") or 0), 2),
                    "change":   round(float(inf.get("regular_market_change_percent") or inf.get("regularMarketChangePercent") or 0), 2),
                }
            except Exception:
                pass
        return macro, "macro"

    return {"error": f"Unknown tool: {name}"}, "unknown"


# ── Research system prompt ────────────────────────────────────────────────────

RESEARCH_SYSTEM = """You are FinBrain's autonomous research intelligence. You have access to real-time tools
to fetch market data, news, fundamentals, and perform web searches. Your task is to conduct thorough,
multi-source financial research and synthesise a comprehensive, actionable report.

Research process:
1. Use tools to gather all relevant data before writing your report
2. Cross-reference technical signals with fundamental data and news sentiment
3. Identify key risks and catalysts
4. Be specific — use actual numbers from the data you retrieved
5. Structure your report clearly with sections

Report format (always use this structure):
## Executive Summary
[2-3 sentence verdict]

## Technical Picture
[Key indicators, trend, momentum, volatility]

## Fundamental Overview
[Valuation, growth, profitability — if applicable]

## Market Sentiment & News
[Recent headlines, sentiment direction, Fear & Greed if relevant]

## Key Risks
[3-5 specific risks with evidence]

## Catalysts to Watch
[3-5 specific near-term catalysts]

## Conclusion
[Balanced assessment — never explicit buy/sell advice]

Be concise, data-driven, and professional. Reference specific numbers you retrieved."""


# ── Route ─────────────────────────────────────────────────────────────────────

@router.post("/research", response_model=ResearchResponse)
async def deep_research(req: ResearchRequest) -> ResearchResponse:
    """Run an agentic research loop with Claude + multi-source data tools.

    Claude autonomously decides which tools to call, runs them, and synthesises
    a structured research report from all gathered data.

    Args:
        req: ResearchRequest with symbol (optional), query, depth.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured")

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
    except ImportError:
        raise HTTPException(status_code=500, detail="anthropic package not installed")

    # Build initial user message
    sym_context = f" for {req.symbol.upper()}" if req.symbol else ""
    max_tokens_map = {"quick": 1500, "standard": 3000, "deep": 4000}
    max_tokens = max_tokens_map.get(req.depth, 3000)

    user_msg = (
        f"Research query{sym_context}: {req.query}\n\n"
        f"Use the available tools to gather comprehensive data, then write a full research report."
    )

    messages = [{"role": "user", "content": user_msg}]
    sources:  list[dict] = []
    data_used: list[str] = []
    total_tokens = 0

    # ── Agentic tool-use loop ─────────────────────────────────────────────────
    for round_num in range(MAX_TOOL_ROUNDS):
        response = client.messages.create(
            model      = "claude-sonnet-4-6",
            max_tokens = max_tokens,
            system     = RESEARCH_SYSTEM,
            tools      = TOOLS,
            messages   = messages,
        )
        total_tokens += response.usage.input_tokens + response.usage.output_tokens

        # If Claude is done (no more tool calls)
        if response.stop_reason == "end_turn":
            break

        # Process tool calls
        tool_results = []
        has_tool_use = False
        for block in response.content:
            if block.type == "tool_use":
                has_tool_use = True
                result, source_label = _execute_tool(block.name, block.input)
                data_used.append(block.name)
                if source_label not in [s.get("type") for s in sources]:
                    sources.append({"type": source_label, "tool": block.name})
                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": block.id,
                    "content":     json.dumps(result, default=str)[:4000],  # cap length
                })

        if not has_tool_use:
            break

        # Continue loop with tool results
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user",      "content": tool_results})

    # ── Extract final text ────────────────────────────────────────────────────
    report_text = ""
    for block in response.content:
        if hasattr(block, "text"):
            report_text += block.text

    if not report_text:
        # Ask for the final report if Claude only did tool calls
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": "Now please write the complete research report based on all the data you collected."})
        final_resp = client.messages.create(
            model      = "claude-sonnet-4-6",
            max_tokens = max_tokens,
            system     = RESEARCH_SYSTEM,
            messages   = messages,
        )
        total_tokens += final_resp.usage.input_tokens + final_resp.usage.output_tokens
        report_text  = "".join(b.text for b in final_resp.content if hasattr(b, "text"))

    return ResearchResponse(
        symbol    = req.symbol,
        query     = req.query,
        report    = report_text,
        sources   = sources,
        data_used = list(set(data_used)),
        model     = "claude-sonnet-4-6",
        tokens    = total_tokens,
    )
