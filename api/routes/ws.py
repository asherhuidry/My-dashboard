"""WebSocket route for real-time price streaming."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import yfinance as yf
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

log = logging.getLogger(__name__)
router = APIRouter()

WATCHLIST = ["AAPL","MSFT","NVDA","TSLA","GOOGL","AMZN","META","JPM","SPY","QQQ",
             "BTC-USD","ETH-USD","SOL-USD","BNB-USD","XRP-USD","GLD","EURUSD=X"]

# In-memory price cache shared across connections
_price_cache: dict[str, dict] = {}


class _Manager:
    def __init__(self) -> None:
        self._clients: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._clients.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self._clients:
            self._clients.remove(ws)

    async def broadcast(self, data: Any) -> None:
        dead = []
        for ws in list(self._clients):
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = _Manager()


async def _fetch_prices() -> dict[str, dict]:
    """Fetch latest prices for the watchlist via yfinance fast_info."""
    prices: dict[str, dict] = {}
    for sym in WATCHLIST:
        try:
            t     = yf.Ticker(sym)
            fi    = t.fast_info
            price = getattr(fi, "last_price", None)
            prev  = getattr(fi, "previous_close", None)
            if price is None:
                price = _price_cache.get(sym, {}).get("price")
            change_pct = round((price - prev) / prev * 100, 2) if price and prev else 0.0
            label = sym.replace("-USD","").replace("=X","")
            prices[sym] = {
                "symbol":     sym,
                "label":      label,
                "price":      round(float(price), 4) if price else None,
                "change_pct": change_pct,
            }
        except Exception:
            if sym in _price_cache:
                prices[sym] = _price_cache[sym]
    return prices


async def price_polling_loop() -> None:
    """Background task: polls prices every 30 s and broadcasts to WebSocket clients."""
    while True:
        try:
            prices = await asyncio.get_event_loop().run_in_executor(None, lambda: asyncio.run(_fetch_prices_sync()))
            _price_cache.update(prices)
            if manager._clients:
                await manager.broadcast({"type": "prices", "data": list(prices.values())})
        except Exception as e:
            log.debug("Price poll error: %s", e)
        await asyncio.sleep(30)


def _fetch_prices_sync() -> dict[str, dict]:
    """Synchronous version for thread executor."""
    import asyncio as _a
    return _a.run(_fetch_prices()) if False else _fetch_prices_thread()


def _fetch_prices_thread() -> dict[str, dict]:
    """Thread-safe price fetch."""
    prices: dict[str, dict] = {}
    for sym in WATCHLIST:
        try:
            t     = yf.Ticker(sym)
            fi    = t.fast_info
            price = getattr(fi, "last_price", None)
            prev  = getattr(fi, "previous_close", None)
            if price is None:
                price = _price_cache.get(sym, {}).get("price")
            change_pct = round((price - prev) / prev * 100, 2) if price and prev else 0.0
            prices[sym] = {
                "symbol":     sym,
                "label":      sym.replace("-USD","").replace("=X",""),
                "price":      round(float(price), 4) if price else None,
                "change_pct": change_pct,
            }
        except Exception:
            if sym in _price_cache:
                prices[sym] = _price_cache[sym]
    return prices


# REST fallback — same data, no WebSocket needed
@router.get("/live-prices")
def live_prices_rest() -> dict:
    """Return cached live prices (REST fallback for clients without WS)."""
    if _price_cache:
        return {"data": list(_price_cache.values()), "source": "cache"}
    # Cold start: fetch synchronously
    data = _fetch_prices_thread()
    _price_cache.update(data)
    return {"data": list(data.values()), "source": "fresh"}


@router.websocket("/ws/prices")
async def websocket_prices(ws: WebSocket) -> None:
    """WebSocket endpoint — streams price updates every 30 s."""
    await manager.connect(ws)
    try:
        # Send current cache immediately on connect
        if _price_cache:
            await ws.send_json({"type": "prices", "data": list(_price_cache.values())})
        # Keep connection alive, echo pings
        while True:
            try:
                msg = await asyncio.wait_for(ws.receive_text(), timeout=60.0)
                if msg == "ping":
                    await ws.send_json({"type": "pong"})
            except asyncio.TimeoutError:
                await ws.send_json({"type": "heartbeat"})
    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception:
        manager.disconnect(ws)
