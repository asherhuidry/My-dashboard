"""FinBrain FastAPI application entry point."""
from __future__ import annotations

import asyncio
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import (
    health, prices, analysis, search, db_stats, evolution,
    graph, chat, screener, ws, predict, backtest, research, data_sources,
    intelligence, neural, experiments, discoveries,
)

log = logging.getLogger(__name__)

app = FastAPI(
    title="FinBrain API",
    description="Autonomous Financial Intelligence System",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router,    prefix="/api")
app.include_router(search.router,    prefix="/api")
app.include_router(prices.router,    prefix="/api")
app.include_router(analysis.router,  prefix="/api")
app.include_router(db_stats.router,  prefix="/api")
app.include_router(evolution.router, prefix="/api")
app.include_router(graph.router,     prefix="/api")
app.include_router(chat.router,      prefix="/api")
app.include_router(screener.router,  prefix="/api")
app.include_router(predict.router,   prefix="/api")
app.include_router(backtest.router,  prefix="/api")
app.include_router(research.router,      prefix="/api")
app.include_router(data_sources.router,  prefix="/api")
app.include_router(intelligence.router,  prefix="/api")
app.include_router(neural.router,        prefix="/api")
app.include_router(experiments.router,   prefix="/api")
app.include_router(discoveries.router,  prefix="/api")
app.include_router(ws.router)        # WebSocket has no /api prefix for ws://


@app.on_event("startup")
async def startup_event() -> None:
    """Launch background price polling task."""
    asyncio.create_task(ws.price_polling_loop())
    log.info("FinBrain API started — price polling active")
