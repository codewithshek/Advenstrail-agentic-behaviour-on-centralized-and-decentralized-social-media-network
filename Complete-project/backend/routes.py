"""HTTP routes for AgentInTheShell inference and analyst dashboard data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from .database import dashboard_summary, get_session, save_analysis

# ``ml_service`` puts ``ml/src`` on the path, so aegis helpers are re-exported
# from there rather than imported directly and creating an import-order trap.
from .ml_service import ArtifactError, service, split_posts
from .models import (
    AccountAnalysisResponse,
    AccountAnalyzeRequest,
    DashboardRecord,
    NetworkAnalysisResponse,
    NetworkAnalyzeRequest,
    TextAnalysisResponse,
    TextAnalyzeRequest,
    ThreadAnalysisResponse,
    ThreadAnalyzeRequest,
    ThreatDashboardResponse,
)

router = APIRouter()
SessionDep = Annotated[Session, Depends(get_session)]
DashboardLimit = Annotated[int, Query(ge=1, le=100)]


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "AgentInTheShell"}


@router.post("/analyze_text", response_model=TextAnalysisResponse)
def analyze_text(
    payload: TextAnalyzeRequest,
    session: SessionDep,
) -> TextAnalysisResponse:
    try:
        result = service.analyze_text(payload.text)
    except ArtifactError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    record = save_analysis(
        session,
        analysis_type="text",
        score=result["score"],
        label=result["label"],
        details={
            "source": payload.source,
            "model": result["model"],
            "artifact_mode": result["artifact_mode"],
            "warning": result["warning"],
        },
    )
    return TextAnalysisResponse(analysis_id=record.id, **result)


@router.post("/analyze_thread", response_model=ThreadAnalysisResponse)
def analyze_thread(
    payload: ThreadAnalyzeRequest,
    session: SessionDep,
) -> ThreadAnalysisResponse:
    posts = payload.posts or split_posts(payload.text or "")
    if not posts:
        raise HTTPException(status_code=422, detail="thread contains no posts")
    try:
        result = service.analyze_thread(posts)
    except ArtifactError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    record = save_analysis(
        session,
        analysis_type="text",
        score=result["peak_score"],
        label=result["label"],
        details={
            "source": payload.source,
            "model": result["model"],
            "artifact_mode": result["artifact_mode"],
            "posts_analyzed": result["posts_analyzed"],
            "flagged_posts": result["flagged_posts"],
            "payload_alerts": len(result["payload_alerts"]),
            "warning": result["warning"],
        },
    )
    return ThreadAnalysisResponse(analysis_id=record.id, **result)


@router.post("/analyze_account", response_model=AccountAnalysisResponse)
def analyze_account(
    payload: AccountAnalyzeRequest,
    session: SessionDep,
) -> AccountAnalysisResponse:
    try:
        result = service.analyze_account(payload.handle, peers=payload.peers)
    except ArtifactError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    record = save_analysis(
        session,
        analysis_type="network",
        score=result["threat_score"],
        label=(
            "adversarial"
            if result["classification"] != "human"
            else "human_benign"
        ),
        details={
            "handle": result["handle"],
            "classification": result["classification"],
            "band": result["band"],
            "data_source": result["data_source"],
            "flagged_nodes": result["network"]["flagged_nodes"],
            "total_nodes": result["network"]["total_nodes"],
            "warning": result["warning"],
        },
    )
    return AccountAnalysisResponse(analysis_id=record.id, **result)


@router.post("/analyze_network", response_model=NetworkAnalysisResponse)
def analyze_network(
    payload: NetworkAnalyzeRequest,
    session: SessionDep,
) -> NetworkAnalysisResponse:
    try:
        result = service.analyze_network(payload)
    except ArtifactError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    label = (
        "adversarial"
        if result["aggregate_score"] >= result["threshold"]
        else "human_benign"
    )
    record = save_analysis(
        session,
        analysis_type="network",
        score=result["aggregate_score"],
        label=label,
        details={
            "model": result["model"],
            "flagged_nodes": result["flagged_nodes"],
            "total_nodes": result["total_nodes"],
            "warning": result["warning"],
        },
    )
    return NetworkAnalysisResponse(analysis_id=record.id, **result)


@router.get("/get_threat_dashboard", response_model=ThreatDashboardResponse)
def get_threat_dashboard(
    session: SessionDep,
    limit: DashboardLimit = 20,
) -> ThreatDashboardResponse:
    summary = dashboard_summary(session, limit=limit)
    recent = [
        DashboardRecord(
            id=row.id,
            analysis_type=row.analysis_type,
            score=row.score,
            label=row.label,
            created_at=row.created_at,
            details=row.details,
        )
        for row in summary.pop("recent")
    ]
    return ThreatDashboardResponse(**summary, recent=recent)


FIXTURES_DIR = Path(__file__).resolve().parents[1] / "datasets" / "fixtures"


@router.get("/architecture_presets")
def get_architecture_presets() -> list[dict[str, Any]]:
    """Return the 4 preloaded Centralized and Decentralized fixture datasets."""
    presets: list[dict[str, Any]] = []
    fixture_order = [
        (
            "centralized_x_hashtag_flood",
            "Centralized: X Hashtag Flood",
            "centralized",
            "Synchronized Hashtag Flood",
        ),
        (
            "centralized_ig_shill_ring",
            "Centralized: Hub Shill Ring",
            "centralized",
            "Algorithmic Retweet Ring",
        ),
        (
            "decentralized_mastodon_bridge_attack",
            "Decentralized: Mastodon Cross-Instance Bridge",
            "decentralized_federated",
            "Cross-Instance Sybil Bridge Attack",
        ),
        (
            "decentralized_a2a_protocol_spoof",
            "Decentralized: A2A Protocol Spoof",
            "decentralized_federated",
            "A2A/MCP Protocol Discovery Spoof",
        ),
    ]
    for fid, name, arch, attack_mode in fixture_order:
        json_file = FIXTURES_DIR / f"{fid}.json"
        if json_file.exists():
            try:
                import json
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                presets.append({
                    "id": fid,
                    "name": name,
                    "architecture": arch,
                    "attack_mode": attack_mode,
                    "metadata": data.get("metadata", {}),
                    "nodes": data.get("nodes", []),
                    "edges": data.get("edges", []),
                    "posts": data.get("posts", []),
                })
            except Exception:
                continue
    return presets
