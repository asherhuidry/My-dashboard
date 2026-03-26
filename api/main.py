"""FinBrain FastAPI application entry point."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import health, prices, analysis, search, db_stats, evolution, graph

app = FastAPI(
    title="FinBrain API",
    description="Autonomous Financial Intelligence System",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
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
