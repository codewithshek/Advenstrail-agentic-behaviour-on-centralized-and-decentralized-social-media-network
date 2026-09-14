"""FastAPI application entry point.

Run from the repository root:
    uvicorn backend.main:app --reload
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .database import create_schema
from .routes import router


@asynccontextmanager
async def lifespan(_: FastAPI):
    create_schema()
    yield


app = FastAPI(
    title="AEGIS-SN Threat Detection API",
    description=(
        "Hybrid text and network analysis for adversarial agentic behavior "
        "in social networks."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

# The dashboard is a static file served separately, so its origin has to be
# allowed explicitly. Both loopback spellings are listed because a browser
# treats localhost and 127.0.0.1 as distinct origins.
_DEFAULT_ORIGINS = (
    "http://localhost:4173,http://127.0.0.1:4173,"
    "http://localhost:5173,http://127.0.0.1:5173,"
    "http://localhost:3000,http://127.0.0.1:3000"
)
origins = [
    value.strip()
    for value in os.getenv("CORS_ORIGINS", _DEFAULT_ORIGINS).split(",")
    if value.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
app.include_router(router)
