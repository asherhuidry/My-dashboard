"""Claude AI chat route — financial analysis assistant."""
from __future__ import annotations

import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

SYSTEM_PROMPT = """You are FinBrain, an elite autonomous financial intelligence system built on cutting-edge ML.
You have access to real-time market data, 74+ technical features computed from OHLCV price history,
and multi-source macro data from FRED, Alpha Vantage, and yfinance.

Your role: Answer financial questions with precision, depth, and actionable insight.

Guidelines:
- Be direct and concise — lead with the key insight
- Use specific numbers when available in the provided context
- Explain technical indicators in plain terms when needed
- Distinguish between technical signals and fundamental factors
- Never give direct buy/sell financial advice — frame everything as analysis
- Format responses with clear structure (bold headers, bullet points where appropriate)
- When context data is provided, reference it specifically

Your capabilities include: technical analysis, pattern recognition, regime detection,
correlation analysis, macro factor impact, feature importance, and signal generation."""


class ChatRequest(BaseModel):
    message: str
    symbol:  str | None = None
    context: dict | None = None


class ChatResponse(BaseModel):
    reply:  str
    model:  str
    tokens: int


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """Send a message to the FinBrain AI assistant."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured")

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
    except ImportError:
        raise HTTPException(status_code=500, detail="anthropic package not installed")

    # Build context block
    context_text = ""
    if req.symbol:
        context_text += f"\n\n**Current analysis context: {req.symbol}**\n"

    if req.context:
        price      = req.context.get("price")
        returns    = req.context.get("returns", {})
        signal     = req.context.get("signal", {})
        features   = req.context.get("features", {})
        name       = req.context.get("name", req.symbol)
        sector     = req.context.get("sector", "")

        if price:
            context_text += f"- Current price: ${price:.2f}\n"
        if name:
            context_text += f"- Company: {name}"
            if sector:
                context_text += f" ({sector})"
            context_text += "\n"

        if returns:
            context_text += f"- Returns: 1D={returns.get('1d','?')}%, 5D={returns.get('5d','?')}%, 21D={returns.get('21d','?')}%\n"

        if signal:
            context_text += f"- Signal: {signal.get('overall','?').upper()} (score: {signal.get('score','?')})\n"

        # Key features
        key_feats = {
            "rsi_14": "RSI(14)",
            "rsi_28": "RSI(28)",
            "macd":   "MACD",
            "adx_14": "ADX(14)",
            "bb_pct": "BB Position",
            "realized_vol_21d": "Vol(21D)",
            "rolling_sharpe_21": "Sharpe(21D)",
            "max_drawdown_63": "Max DD(63D)",
        }
        feat_lines = []
        for k, label in key_feats.items():
            v = features.get(k)
            if v is not None:
                feat_lines.append(f"  {label}: {v:.3f}")
        if feat_lines:
            context_text += "- Key indicators:\n" + "\n".join(feat_lines) + "\n"

    user_message = req.message
    if context_text:
        user_message = f"{context_text}\n\nUser question: {req.message}"

    try:
        response = client.messages.create(
            model      = "claude-haiku-4-5-20251001",
            max_tokens = 1024,
            system     = SYSTEM_PROMPT,
            messages   = [{"role": "user", "content": user_message}],
        )
        reply  = response.content[0].text
        tokens = response.usage.input_tokens + response.usage.output_tokens
        return ChatResponse(reply=reply, model=response.model, tokens=tokens)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Claude API error: {e}")
