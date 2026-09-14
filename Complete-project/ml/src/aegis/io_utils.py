"""
aegis.io_utils
==============

Frame persistence, content hashing and the provenance manifest.

Two design decisions worth defending
------------------------------------
1.  **Parquet preferred, CSV.gz fallback.** Parquet preserves dtypes (critical:
    ``label`` must stay ``int8``, ``created_at`` must stay tz-aware datetime)
    and is 5-10x smaller. But it needs ``pyarrow``, and a marker's / reviewer's
    fresh environment may not have it. :func:`save_frame` therefore degrades to
    gzipped CSV and records the format it actually used, so
    :func:`load_frame` finds the file either way. Nothing in the pipeline
    hard-fails on a missing optional dependency.

2.  **Content hashing over row counts.** ``manifest.json`` stores a SHA-256 of
    the canonicalised frame. Row counts collide trivially; a hash lets you
    prove in the write-up that the table you evaluated on is the table you
    documented — including proving that a *synthetic fallback* was or was not
    substituted for a real corpus.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import Paths, get_logger

log = get_logger("aegis.io")

# Provenance vocabulary. Kept as plain strings (not an Enum) so the manifest
# stays human-diffable in git.
PROV_REAL = "REAL"
PROV_SYNTHETIC_FALLBACK = "SYNTHETIC_FALLBACK"
PROV_GENERATED = "GENERATED"          # our own synthetic campaign — intended, not a fallback
PROV_PARTIAL = "PARTIAL"              # real data, but truncated by smoke_test / row cap

_ALL_PROVENANCE = (PROV_REAL, PROV_SYNTHETIC_FALLBACK, PROV_GENERATED, PROV_PARTIAL)


# --------------------------------------------------------------------------- #
# Optional-dependency probes
# --------------------------------------------------------------------------- #
def has_module(name: str) -> bool:
    """True if ``name`` is importable, without leaving it in ``sys.modules``."""
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


HAS_PARQUET = has_module("pyarrow") or has_module("fastparquet")


# --------------------------------------------------------------------------- #
# Hashing
# --------------------------------------------------------------------------- #
def frame_fingerprint(df: pd.DataFrame, *, columns: Optional[Iterable[str]] = None) -> str:
    """
    Order-independent SHA-256 of a frame's content.

    Rows are hashed individually then the *sorted* row digests are folded
    together, so a reshuffle does not change the fingerprint. That is what we
    want: shuffling is a legitimate pipeline step, mutating a value is not.
    """
    cols = list(columns) if columns else [c for c in df.columns]
    cols = [c for c in cols if c in df.columns]
    if not cols or df.empty:
        return hashlib.sha256(b"__empty__").hexdigest()

    subset = df[cols]
    row_digests: List[bytes] = []
    for row in subset.itertuples(index=False, name=None):
        payload = "\x1f".join("" if v is None else str(v) for v in row)
        row_digests.append(hashlib.blake2b(payload.encode("utf-8"), digest_size=16).digest())
    row_digests.sort()

    outer = hashlib.sha256()
    outer.update(("|".join(cols)).encode("utf-8"))
    for digest in row_digests:
        outer.update(digest)
    return outer.hexdigest()


def file_sha256(path: Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# Frame persistence
# --------------------------------------------------------------------------- #
def save_frame(
    df: pd.DataFrame,
    path: Path,
    *,
    prefer_parquet: bool = True,
    index: bool = False,
) -> Path:
    """
    Write ``df`` to ``path``, choosing parquet when possible.

    ``path`` may be given with or without a suffix; the effective suffix is
    decided here. Returns the path actually written.
    """
    path = Path(path)
    stem_path = path.with_suffix("")
    path.parent.mkdir(parents=True, exist_ok=True)

    if prefer_parquet and HAS_PARQUET:
        target = stem_path.with_suffix(".parquet")
        # Object columns containing mixed types break the arrow converter;
        # stringify them rather than losing the whole write.
        safe = df.copy()
        for col in safe.columns:
            if safe[col].dtype == object:
                non_null = safe[col].dropna()
                if not non_null.empty and non_null.map(type).nunique() > 1:
                    safe[col] = safe[col].map(lambda v: v if v is None else str(v))
        safe.to_parquet(target, index=index)
    else:
        if prefer_parquet and not HAS_PARQUET:
            log.debug("pyarrow unavailable — writing gzipped CSV instead of parquet.")
        target = stem_path.with_suffix(".csv.gz")
        df.to_csv(target, index=index, compression="gzip")

    log.info("wrote %-38s rows=%-8d cols=%-3d (%.1f KB)",
             target.name, len(df), df.shape[1], target.stat().st_size / 1024)
    return target


def resolve_frame_path(path: Path) -> Optional[Path]:
    """Find whichever serialisation of ``path`` exists (parquet | csv.gz | csv)."""
    path = Path(path)
    if path.exists() and path.is_file():
        return path
    stem = path.with_suffix("")
    for suffix in (".parquet", ".csv.gz", ".csv", ".jsonl"):
        candidate = stem.with_suffix(suffix)
        if candidate.exists():
            return candidate
    # ".csv.gz" has two suffixes, so with_suffix() above mangles "x.csv" -> "x.gz".
    for suffix in (".parquet", ".csv.gz", ".csv", ".jsonl"):
        candidate = Path(str(stem) + suffix)
        if candidate.exists():
            return candidate
    return None


def load_frame(path: Path, **read_kwargs) -> pd.DataFrame:
    """Read a frame written by :func:`save_frame`, format-agnostically."""
    resolved = resolve_frame_path(path)
    if resolved is None:
        raise FileNotFoundError(
            f"No parquet/csv.gz/csv artefact found for {path}. "
            f"Run 01_data_ingestion_and_synthetic_gen.ipynb first."
        )
    name = resolved.name
    if name.endswith(".parquet"):
        return pd.read_parquet(resolved, **read_kwargs)
    if name.endswith(".jsonl"):
        return pd.read_json(resolved, lines=True, **read_kwargs)
    return pd.read_csv(resolved, **read_kwargs)


def save_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> Path:
    """Line-delimited JSON — the right format for the agent conversation logs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            n += 1
    log.info("wrote %-38s records=%d", path.name, n)
    return path


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def save_json(obj: Any, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    return path


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return str(value)


# --------------------------------------------------------------------------- #
# Provenance record
# --------------------------------------------------------------------------- #
@dataclass
class DatasetRecord:
    """
    One row of the provenance ledger.

    ``provenance`` is the field that keeps the project honest. If a reviewer
    asks "did you actually evaluate on TwiBot-24 or on a stub?", this answers
    it definitively, and :func:`assert_real_data` can be dropped into the
    notebook that produces the final table to make an accidental
    synthetic-fallback publication impossible.
    """

    name: str
    provenance: str
    source: str
    rows: int
    columns: List[str]
    sha256: str
    retrieved_at: str
    artefact: Optional[str] = None
    era: Optional[str] = None
    label_balance: Optional[Dict[str, int]] = None
    note: str = ""
    env: Optional[Dict[str, str]] = None

    def __post_init__(self) -> None:
        if self.provenance not in _ALL_PROVENANCE:
            raise ValueError(
                f"provenance must be one of {_ALL_PROVENANCE}, got {self.provenance!r}"
            )

    @property
    def is_real(self) -> bool:
        return self.provenance in (PROV_REAL, PROV_PARTIAL)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_record(
    name: str,
    df: pd.DataFrame,
    *,
    provenance: str,
    source: str,
    era: Optional[str] = None,
    artefact: Optional[Path] = None,
    label_col: Optional[str] = "label",
    note: str = "",
) -> DatasetRecord:
    balance = None
    if label_col and label_col in df.columns:
        balance = {str(k): int(v) for k, v in df[label_col].value_counts().items()}
    return DatasetRecord(
        name=name,
        provenance=provenance,
        source=source,
        rows=int(len(df)),
        columns=[str(c) for c in df.columns],
        sha256=frame_fingerprint(df),
        retrieved_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        artefact=str(artefact) if artefact else None,
        era=era,
        label_balance=balance,
        note=note,
        env={
            "python": platform.python_version(),
            "platform": platform.platform(terse=True),
            "pandas": pd.__version__,
        },
    )


def register(paths: Paths, record: DatasetRecord) -> DatasetRecord:
    """Append/replace ``record`` in ``data/manifest.json``."""
    from .config import GLYPHS, write_manifest_entry

    write_manifest_entry(paths, record.name, record.to_dict())
    tag = (
        GLYPHS["ok"] if record.is_real
        else (GLYPHS["gen"] if record.provenance == PROV_GENERATED else GLYPHS["warn"])
    )
    log.info(
        "%s manifest[%s] provenance=%s source=%s rows=%d",
        tag, record.name, record.provenance, record.source, record.rows,
    )
    return record


def assert_real_data(paths: Paths, names: Iterable[str], *, strict: bool = True) -> pd.DataFrame:
    """
    Guard for the final results notebook.

    Call this immediately before producing any table or figure that will be
    quoted as a result. With ``strict=True`` it raises if any named dataset was
    served from a synthetic fallback.
    """
    from .config import read_manifest

    manifest = read_manifest(paths).get("datasets", {})
    rows = []
    offenders = []
    for name in names:
        entry = manifest.get(name)
        if entry is None:
            offenders.append(f"{name}: absent from manifest")
            rows.append({"dataset": name, "provenance": "MISSING", "rows": 0})
            continue
        prov = entry.get("provenance")
        rows.append({"dataset": name, "provenance": prov, "rows": entry.get("rows", 0)})
        if prov == PROV_SYNTHETIC_FALLBACK:
            offenders.append(f"{name}: served from SYNTHETIC_FALLBACK")

    report = pd.DataFrame(rows)
    if offenders:
        message = (
            "Results integrity check FAILED — these inputs are not real data:\n  - "
            + "\n  - ".join(offenders)
            + "\n\nMetrics computed on stubs must never be reported as results. "
              "Provide credentials / dataset access, re-run notebook 01, then retry."
        )
        if strict:
            raise AssertionError(message)
        log.error(message)
    else:
        log.info("Results integrity check passed for %d datasets.", len(report))
    return report


# --------------------------------------------------------------------------- #
# Misc
# --------------------------------------------------------------------------- #
def cap_rows(
    df: pd.DataFrame,
    limit: Optional[int],
    *,
    stratify_col: Optional[str] = "label",
    seed: int = 42,
) -> Tuple[pd.DataFrame, bool]:
    """
    Truncate to ``limit`` rows, preserving class balance where possible.

    Returns ``(frame, was_truncated)`` so the caller can downgrade the
    provenance to ``PARTIAL`` — a truncated real corpus is still real, but the
    distinction matters when comparing against published numbers.
    """
    if limit is None or len(df) <= limit:
        return df, False

    original = len(df)
    if stratify_col and stratify_col in df.columns and df[stratify_col].nunique() > 1:
        # NB: deliberately NOT `groupby(...).apply(lambda g: g.sample(...))`.
        # Since pandas 2.2 the grouping column is excluded from the frame handed
        # to `apply`, so that idiom silently returns a frame with no `label`
        # column — the corpus survives, the target does not, and the failure
        # only surfaces several cells later. Sampling positional indices keeps
        # every column and is faster besides.
        rng = np.random.default_rng(seed)
        frac = limit / original
        chosen: List[np.ndarray] = []
        for positions in df.groupby(stratify_col, sort=False, observed=True).indices.values():
            take = min(len(positions), max(1, int(round(len(positions) * frac))))
            chosen.append(rng.choice(positions, size=take, replace=False))
        selected = np.concatenate(chosen)
        rng.shuffle(selected)
        out = df.iloc[selected[:limit]].reset_index(drop=True)
    else:
        out = df.sample(n=limit, random_state=seed).reset_index(drop=True)

    log.info("cap_rows: %d -> %d rows (smoke test)", original, len(out))
    return out, True


def human_bytes(n: int) -> str:
    step = 1024.0
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < step:
            return f"{value:.1f} {unit}"
        value /= step
    return f"{value:.1f} PB"


def directory_size(path: Path) -> int:
    path = Path(path)
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


__all__ = [
    "PROV_REAL", "PROV_SYNTHETIC_FALLBACK", "PROV_GENERATED", "PROV_PARTIAL",
    "HAS_PARQUET", "has_module",
    "frame_fingerprint", "file_sha256",
    "save_frame", "load_frame", "resolve_frame_path",
    "save_jsonl", "load_jsonl", "save_json",
    "DatasetRecord", "build_record", "register", "assert_real_data",
    "cap_rows", "human_bytes", "directory_size",
]
