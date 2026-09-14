"""SQLAlchemy persistence with SQLite development and PostgreSQL support."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import (
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    func,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = f"sqlite:///{(ROOT / 'data' / 'aegis_api.db').as_posix()}"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_DB)

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class AnalysisRecord(Base):
    __tablename__ = "analysis_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    analysis_type: Mapped[str] = mapped_column(String(20), index=True)
    score: Mapped[float] = mapped_column(Float)
    label: Mapped[str] = mapped_column(String(30), index=True)
    details_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )

    @property
    def details(self) -> dict[str, Any]:
        try:
            value = json.loads(self.details_json)
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}


def create_schema() -> None:
    ROOT.joinpath("data").mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(engine)


def get_session() -> Iterator[Session]:
    with SessionLocal() as session:
        yield session


def save_analysis(
    session: Session,
    *,
    analysis_type: str,
    score: float,
    label: str,
    details: dict[str, Any],
) -> AnalysisRecord:
    record = AnalysisRecord(
        analysis_type=analysis_type,
        score=float(score),
        label=label,
        details_json=json.dumps(details, default=str),
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    return record


def dashboard_summary(session: Session, *, limit: int = 20) -> dict[str, Any]:
    total = session.scalar(select(func.count()).select_from(AnalysisRecord)) or 0
    text_count = session.scalar(
        select(func.count()).select_from(AnalysisRecord).where(
            AnalysisRecord.analysis_type == "text"
        )
    ) or 0
    network_count = session.scalar(
        select(func.count()).select_from(AnalysisRecord).where(
            AnalysisRecord.analysis_type == "network"
        )
    ) or 0
    adversarial = session.scalar(
        select(func.count()).select_from(AnalysisRecord).where(
            AnalysisRecord.label == "adversarial"
        )
    ) or 0
    mean_score = session.scalar(select(func.avg(AnalysisRecord.score))) or 0.0
    recent = session.scalars(
        select(AnalysisRecord)
        .order_by(AnalysisRecord.created_at.desc(), AnalysisRecord.id.desc())
        .limit(limit)
    ).all()
    return {
        "total_analyses": int(total),
        "text_analyses": int(text_count),
        "network_analyses": int(network_count),
        "adversarial_findings": int(adversarial),
        "mean_score": float(mean_score),
        "recent": recent,
    }
