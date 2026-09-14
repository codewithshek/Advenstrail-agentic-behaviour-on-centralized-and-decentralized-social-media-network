"""
aegis.local_store
=================

Discovery and tolerant reading of hand-placed corpora in ``datasets/<name>/``.

This module exists because of one operational reality: **the raw download never
matches the spec.** A user who fetches TweepFake from Kaggle gets a zip; from
Zenodo they get a nested folder; a mirror renames ``account.type`` to
``account_type``; WildJailbreak ships TSV while WildGuard ships parquet. If the
loader is strict about any of that, the user has to hand-massage nine corpora
before the pipeline runs once — which in practice means it never runs.

So the contract here is deliberately forgiving:

*   **Archives are extracted in place**, once, with a ``.aegis_extracted``
    sentinel so re-runs are instant.
*   **Discovery is recursive and case-insensitive**, driven by glob patterns
    from ``ml/configs/default.yaml`` (``discovery:`` per dataset).
*   **Format is sniffed, not assumed.** Delimiter detection for csv/tsv/txt,
    JSON-vs-JSONL detection by peeking at the first non-blank byte.
*   **Columns are matched through an alias table** (``column_aliases:`` in the
    config), normalising case, dots, underscores and spaces — so
    ``account.type``, ``Account_Type`` and ``accounttype`` all resolve.

Everything is stdlib + pandas. No network, no optional heavy dependency.

Security note
-------------
:func:`extract_archives` refuses entries whose resolved path escapes the target
directory (Zip-Slip / tar path traversal). These archives come from third-party
mirrors; treating them as trusted input would be a real vulnerability, not a
theoretical one.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import os
import re
import shutil
import tarfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import pandas as pd

from .config import get_logger

log = get_logger("aegis.local")

TABULAR_SUFFIXES = (".csv", ".tsv", ".txt", ".json", ".jsonl", ".ndjson", ".parquet")
ARCHIVE_SUFFIXES = (".zip", ".tar.gz", ".tgz", ".tar", ".gz")
_EXTRACT_SENTINEL = ".aegis_extracted"

# Files that are never data, however they are named.
_IGNORE_NAMES = {
    "readme.md", "readme.txt", "readme", "license", "license.txt", "licence",
    ".gitkeep", ".ds_store", "thumbs.db", _EXTRACT_SENTINEL,
}
_IGNORE_DIR_PARTS = {"__macosx", ".git", ".ipynb_checkpoints", "__pycache__"}


# --------------------------------------------------------------------------- #
# Column-name normalisation
# --------------------------------------------------------------------------- #
def canon(name: object) -> str:
    """
    Canonical form of a column name for alias matching.

    ``"Account.Type "`` -> ``"accounttype"``. Aggressive on purpose: the only
    thing we care about is whether two names *mean* the same field, and mirrors
    differ exclusively in punctuation and case.
    """
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


def resolve_column(
    df: pd.DataFrame, aliases: Sequence[str], *, required: bool = False
) -> Optional[str]:
    """
    Return the real column in ``df`` matching any name in ``aliases``.

    Three passes, most-precise first. The pass structure matters more than it
    looks — a naive "is either string a substring of the other" rule resolves
    the alias ``source_id`` to a column literally named ``id``, which silently
    produces a self-loop-only interaction graph and a GNN that trains to 0.99
    accuracy on nothing. Learned that the hard way; hence the length guards.

    1.  **Exact** match on canonical form. ``account.type`` == ``Account_Type``.
    2.  **Alias contained in column name** (``followers`` -> ``followers_count``).
        Requires the alias to be >= 5 chars, so the 4-char alias ``text`` cannot
        capture ``human_text``.
    3.  **Column name contained in alias** (``created`` -> alias ``created_at``).
        Requires >= 5 chars *and* >= 60% length overlap, which admits
        ``created``/``created_at`` while rejecting ``id``/``source_id``.
    """
    lookup: Dict[str, str] = {}
    for column in df.columns:
        lookup.setdefault(canon(column), str(column))

    # Pass 1 — exact on canonical form.
    for alias in aliases:
        hit = lookup.get(canon(alias))
        if hit is not None:
            return hit

    # Pass 2 — alias is a substring of the column name.
    for alias in aliases:
        target = canon(alias)
        if len(target) < 5:
            continue
        for key, column in lookup.items():
            if target in key:
                return column

    # Pass 3 — column name is a substring of the alias (abbreviated headers).
    for alias in aliases:
        target = canon(alias)
        for key, column in lookup.items():
            if (
                len(key) >= 5
                and key in target
                and len(key) / max(len(target), 1) >= 0.6
            ):
                return column

    if required:
        raise KeyError(
            f"None of {list(aliases)} found. Available columns: {list(df.columns)[:40]}"
        )
    return None


def resolve_columns(
    df: pd.DataFrame, alias_table: Dict[str, Sequence[str]]
) -> Dict[str, Optional[str]]:
    """Resolve a whole ``{field: [aliases...]}`` table in one call."""
    return {field: resolve_column(df, aliases) for field, aliases in alias_table.items()}


# --------------------------------------------------------------------------- #
# Archive extraction
# --------------------------------------------------------------------------- #
def _is_within(base: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def _safe_extract_zip(archive: Path, dest: Path) -> int:
    count = 0
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            out_path = dest / member.filename
            if not _is_within(dest, out_path):
                log.warning("Refusing path-traversal entry %r in %s", member.filename, archive.name)
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, out_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            count += 1
    return count


def _safe_extract_tar(archive: Path, dest: Path) -> int:
    count = 0
    mode = "r:gz" if archive.name.endswith((".tar.gz", ".tgz")) else "r:*"
    with tarfile.open(archive, mode) as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            out_path = dest / member.name
            if not _is_within(dest, out_path):
                log.warning("Refusing path-traversal entry %r in %s", member.name, archive.name)
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            extracted = tf.extractfile(member)
            if extracted is None:
                continue
            with extracted as src, out_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            count += 1
    return count


def extract_archives(root: Path, *, enabled: bool = True) -> List[Path]:
    """
    Extract every archive under ``root`` in place, once.

    A ``<archive>.aegis_extracted`` marker prevents repeat work, so the second
    notebook run costs nothing. Returns the archives that were processed.
    """
    root = Path(root)
    if not enabled or not root.exists():
        return []

    processed: List[Path] = []
    for archive in sorted(root.rglob("*")):
        if not archive.is_file():
            continue
        name = archive.name.lower()
        if not name.endswith(ARCHIVE_SUFFIXES):
            continue
        marker = archive.with_name(archive.name + _EXTRACT_SENTINEL)
        if marker.exists():
            continue

        dest = archive.parent / re.sub(
            r"(\.tar\.gz|\.tgz|\.tar|\.zip|\.gz)$", "", archive.name, flags=re.IGNORECASE
        )
        try:
            if name.endswith(".zip"):
                dest.mkdir(parents=True, exist_ok=True)
                n = _safe_extract_zip(archive, dest)
            elif name.endswith((".tar.gz", ".tgz", ".tar")):
                dest.mkdir(parents=True, exist_ok=True)
                n = _safe_extract_tar(archive, dest)
            elif name.endswith(".gz"):
                # A bare .gz is a single compressed file, not a container.
                # pandas reads .csv.gz directly, so only expand the odd cases.
                if re.search(r"\.(csv|tsv|json|jsonl|txt)\.gz$", name):
                    marker.write_text("skipped: pandas reads this directly\n", encoding="utf-8")
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                with gzip.open(archive, "rb") as src, dest.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                n = 1
            else:
                continue
            marker.write_text(f"extracted {n} files to {dest.name}\n", encoding="utf-8")
            log.info("extracted %s -> %s/ (%d files)", archive.name, dest.name, n)
            processed.append(archive)
        except Exception as exc:  # noqa: BLE001 — a bad archive must not kill the run
            log.warning("Could not extract %s: %s", archive.name, exc)
            marker.write_text(f"FAILED: {exc}\n", encoding="utf-8")
    return processed


# --------------------------------------------------------------------------- #
# Reading INSIDE archives, without extracting them
#
# `extract_archives` is the right tool for a 20 MB tsv. It is the wrong tool for
# the three large Bot Repository releases: cresci-2017.csv.zip is 466 MB of
# nested per-group zips that expand to 2.3 GB of tweets, and cresci-2015 adds
# another 810 MB. Those two are read in place instead — one pass over the outer
# container, each group's inner zip held in memory (208 MB at the worst, the
# genuine_accounts group) while pandas streams the CSVs out of it in chunks.
# --------------------------------------------------------------------------- #
def iter_archive_members(
    container: Path, *, suffix: str = ".zip"
) -> Iterator[Tuple[str, bytes]]:
    """
    Yield ``(member_name, raw_bytes)`` for members of ``container`` ending in
    ``suffix``, without writing anything to disk.

    Handles zip and tar.gz containers, which is what the two Cresci releases
    are (``cresci-2017.csv.zip`` holds ten ``<group>.csv.zip``;
    ``cresci-2015.csv.tar.gz`` holds five). Members are yielded in sorted order
    so a run is reproducible, and ``__MACOSX`` resource forks are skipped —
    they are zero-length AppleDouble stubs that would otherwise be mistaken for
    a group.
    """
    container = Path(container)
    name = container.name.lower()
    wanted = suffix.lower()

    if name.endswith(".zip"):
        with zipfile.ZipFile(container) as zf:
            members = sorted(
                m.filename for m in zf.infolist()
                if not m.is_dir()
                and m.filename.lower().endswith(wanted)
                and "__macosx" not in m.filename.lower()
            )
            for member in members:
                with zf.open(member) as handle:
                    yield member, handle.read()
        return

    if name.endswith((".tar.gz", ".tgz", ".tar")):
        mode = "r:gz" if name.endswith((".tar.gz", ".tgz")) else "r:*"
        with tarfile.open(container, mode) as tf:
            members = sorted(
                (m for m in tf.getmembers()
                 if m.isfile()
                 and m.name.lower().endswith(wanted)
                 and "__macosx" not in m.name.lower()),
                key=lambda m: m.name,
            )
            for member in members:
                extracted = tf.extractfile(member)
                if extracted is None:
                    continue
                with extracted as handle:
                    yield member.name, handle.read()
        return

    raise ValueError(f"{container.name} is not a zip or tar container")


def open_zip_bytes(blob: bytes) -> zipfile.ZipFile:
    """Wrap raw zip bytes so members can be streamed without touching disk."""
    return zipfile.ZipFile(io.BytesIO(blob))


def zip_member(archive: zipfile.ZipFile, filename: str) -> Optional[str]:
    """
    Resolve a bare filename to its full member path inside ``archive``.

    Members arrive as ``genuine_accounts.csv/users.csv`` in cresci-2017 but as
    a bare ``users.csv`` in cresci-2015, so matching on the tail is the only
    rule that works for both.
    """
    target = filename.lower()
    for member in archive.namelist():
        low = member.lower()
        if "__macosx" in low or low.endswith("/"):
            continue
        if low == target or low.endswith("/" + target):
            return member
    return None


def list_archives(root: Path) -> List[Path]:
    """Every archive under ``root``, recursively, sorted."""
    root = Path(root)
    if not root.exists():
        return []
    return sorted(
        path for path in root.rglob("*")
        if path.is_file()
        and not _ignored(path)
        and path.name.lower().endswith(ARCHIVE_SUFFIXES)
    )


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def _ignored(path: Path) -> bool:
    if path.name.lower() in _IGNORE_NAMES:
        return True
    if path.name.endswith(_EXTRACT_SENTINEL):
        return True
    lowered = {part.lower() for part in path.parts}
    return bool(lowered & _IGNORE_DIR_PARTS)


def list_data_files(
    root: Path, *, suffixes: Sequence[str] = TABULAR_SUFFIXES
) -> List[Path]:
    """Every plausible data file under ``root``, recursively, sorted."""
    root = Path(root)
    if not root.exists():
        return []
    allowed = tuple(s.lower() for s in suffixes)
    out = []
    for path in root.rglob("*"):
        if not path.is_file() or _ignored(path):
            continue
        name = path.name.lower()
        if name.endswith(allowed) or name.endswith(tuple(s + ".gz" for s in allowed)):
            out.append(path)
    return sorted(out)


def find_files(
    root: Path,
    patterns: Sequence[str],
    *,
    recursive: bool = True,
    suffixes: Sequence[str] = TABULAR_SUFFIXES,
) -> List[Path]:
    """
    Case-insensitive glob over ``root``.

    Patterns are matched against both the bare filename and the path relative to
    ``root``, so ``"**/users.csv"`` and ``"users.csv"`` both work regardless of
    how deeply the download nested things.
    """
    import fnmatch

    candidates = list_data_files(root, suffixes=suffixes) if recursive else [
        p for p in Path(root).glob("*") if p.is_file() and not _ignored(p)
    ]
    if not candidates:
        return []

    root = Path(root)
    matched: List[Path] = []
    for pattern in patterns:
        low = pattern.lower()
        bare = low.split("/")[-1]
        for path in candidates:
            if path in matched:
                continue
            rel = str(path.relative_to(root)).replace(os.sep, "/").lower()
            name = path.name.lower()
            if (
                fnmatch.fnmatch(rel, low)
                or fnmatch.fnmatch(name, low)
                or fnmatch.fnmatch(name, bare)
                or fnmatch.fnmatch(rel, f"*/{bare}")
            ):
                matched.append(path)
    return matched


# --------------------------------------------------------------------------- #
# Tolerant reading
# --------------------------------------------------------------------------- #
def _open_text(path: Path):
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def sniff_delimiter(path: Path, *, default: str = ",") -> str:
    """Detect the delimiter of a delimited text file from its first few KB."""
    if path.name.lower().replace(".gz", "").endswith(".tsv"):
        return "\t"
    try:
        with _open_text(path) as fh:
            sample = fh.read(64 * 1024)
        if not sample:
            return default
        try:
            return csv.Sniffer().sniff(sample, delimiters=",\t;|").delimiter
        except csv.Error:
            # Fall back to "whichever candidate appears most on line 1".
            first = sample.splitlines()[0] if sample.splitlines() else ""
            counts = {d: first.count(d) for d in (",", "\t", ";", "|")}
            best = max(counts, key=counts.get)
            return best if counts[best] > 0 else default
    except Exception:  # noqa: BLE001
        return default


def _is_json_lines(path: Path) -> bool:
    """JSON array vs JSON-lines, decided by the first non-whitespace byte."""
    try:
        with _open_text(path) as fh:
            while True:
                chunk = fh.read(1)
                if not chunk:
                    return False
                if chunk.isspace():
                    continue
                return chunk != "["
    except Exception:  # noqa: BLE001
        return True


def read_any(
    path: Path,
    *,
    nrows: Optional[int] = None,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Read one data file into a DataFrame, whatever its format.

    Tries hard rather than failing: a malformed CSV falls back to the Python
    engine with ``on_bad_lines="skip"``, because a handful of broken rows in a
    3 GB mirror should cost you those rows, not the corpus.
    """
    path = Path(path)
    name = path.name.lower()
    base = name[:-3] if name.endswith(".gz") else name

    if base.endswith(".parquet"):
        frame = pd.read_parquet(path, columns=list(columns) if columns else None)
        return frame.head(nrows) if nrows else frame

    if base.endswith((".jsonl", ".ndjson")):
        return pd.read_json(path, lines=True, nrows=nrows) if nrows else pd.read_json(
            path, lines=True
        )

    if base.endswith(".json"):
        if _is_json_lines(path):
            try:
                return (
                    pd.read_json(path, lines=True, nrows=nrows)
                    if nrows
                    else pd.read_json(path, lines=True)
                )
            except ValueError:
                pass
        with _open_text(path) as fh:
            payload = json.load(fh)
        if isinstance(payload, dict):
            # Either {"records": [...]} or a dict-of-columns / dict-of-records.
            for key in ("data", "records", "rows", "items"):
                if isinstance(payload.get(key), list):
                    payload = payload[key]
                    break
            else:
                try:
                    frame = pd.DataFrame.from_dict(payload, orient="index")
                    frame.index.name = "id"
                    frame = frame.reset_index()
                    return frame.head(nrows) if nrows else frame
                except Exception:  # noqa: BLE001
                    payload = [payload]
        frame = pd.json_normalize(payload)
        return frame.head(nrows) if nrows else frame

    # Delimited text
    delimiter = sniff_delimiter(path)
    common: Dict[str, Any] = dict(
        sep=delimiter,
        nrows=nrows,
        usecols=list(columns) if columns else None,
        encoding="utf-8",
        encoding_errors="replace",
        low_memory=False,
    )
    try:
        return pd.read_csv(path, **common)
    except Exception as first_error:  # noqa: BLE001
        log.debug("strict read failed for %s (%s) — retrying leniently", path.name, first_error)
        lenient = dict(common)
        lenient.pop("low_memory", None)
        lenient.update(engine="python", on_bad_lines="skip", quoting=csv.QUOTE_MINIMAL)
        return pd.read_csv(path, **lenient)


def read_many(
    paths: Iterable[Path],
    *,
    nrows_per_file: Optional[int] = None,
    add_source_column: bool = True,
    max_files: Optional[int] = None,
) -> pd.DataFrame:
    """
    Concatenate several files with heterogeneous schemas.

    ``add_source_column`` keeps ``__source_file`` and ``__source_dir``, which
    Cresci needs (its label lives in the directory name) and which makes any
    later data-quality question answerable.
    """
    frames: List[pd.DataFrame] = []
    used = 0
    for path in paths:
        if max_files is not None and used >= max_files:
            break
        try:
            frame = read_any(path, nrows=nrows_per_file)
        except Exception as exc:  # noqa: BLE001
            log.warning("skipping unreadable %s: %s", path, exc)
            continue
        if frame.empty:
            continue
        if add_source_column:
            frame["__source_file"] = path.name
            frame["__source_dir"] = path.parent.name
        frames.append(frame)
        used += 1

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


# --------------------------------------------------------------------------- #
# Dataset-level view of the drop-zone
# --------------------------------------------------------------------------- #
@dataclass
class LocalDataset:
    """What we found in ``datasets/<name>/`` for one corpus."""

    name: str
    root: Path
    exists: bool
    files: List[Path] = field(default_factory=list)
    groups: Dict[str, List[Path]] = field(default_factory=dict)
    total_bytes: int = 0
    # Archives that were deliberately NOT expanded (`extract_archives: false`).
    # The three large Cresci/Caverlee releases are read in place, so for those
    # the container IS the corpus and `files` is legitimately empty.
    archives: List[Path] = field(default_factory=list)
    archive_bytes: int = 0

    @property
    def available(self) -> bool:
        return self.exists and bool(self.files or self.archives)

    @property
    def n_files(self) -> int:
        return len(self.files)

    def group(self, key: str, *, fallback_to_any: bool = True) -> List[Path]:
        """Files for a named discovery group (``users``, ``train``, ...)."""
        hits = self.groups.get(key) or []
        if hits:
            return hits
        if fallback_to_any:
            return self.groups.get("any") or self.files
        return []

    def summary(self) -> Dict[str, Any]:
        return {
            "dataset": self.name,
            "available": self.available,
            "n_files": self.n_files,
            "size": _human(self.total_bytes),
            "groups": {k: len(v) for k, v in self.groups.items() if v},
            "archives": [a.name for a in self.archives],
            "archive_size": _human(self.archive_bytes),
            "root": str(self.root),
        }

    def __repr__(self) -> str:  # pragma: no cover
        state = "AVAILABLE" if self.available else ("EMPTY" if self.exists else "MISSING")
        return (
            f"<LocalDataset {self.name} {state} files={self.n_files} "
            f"size={_human(self.total_bytes)} archives={len(self.archives)}>"
        )


def _human(n: int) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} PB"


def scan_dataset(
    name: str,
    spec: Dict[str, Any],
    *,
    repo_root: Path,
    manual_root: str = "datasets",
    extract: bool = True,
    recursive: bool = True,
) -> LocalDataset:
    """
    Inspect the drop-zone folder for one dataset.

    ``spec`` is that dataset's block from ``default.yaml``; its ``manual_dir``
    and ``discovery`` keys drive where we look and how files are bucketed.
    """
    rel = spec.get("manual_dir") or f"{manual_root}/{name}"
    root = (Path(repo_root) / rel).resolve()

    if not root.exists():
        return LocalDataset(name=name, root=root, exists=False)

    # `extract_archives: false` in a dataset block means "the loader reads the
    # container in place". Expanding cresci-2017.csv.zip would cost ~2.8 GB of
    # disk for files that are streamed out of the zip in one pass anyway.
    expand = extract and bool(spec.get("extract_archives", True))
    extract_archives(root, enabled=expand)

    files = list_data_files(root)
    groups: Dict[str, List[Path]] = {}
    for key, patterns in (spec.get("discovery") or {}).items():
        groups[key] = find_files(root, patterns, recursive=recursive)
    if "any" not in groups:
        groups["any"] = files

    archives = list_archives(root) if not expand else []

    return LocalDataset(
        name=name,
        root=root,
        exists=True,
        files=files,
        groups=groups,
        total_bytes=sum(f.stat().st_size for f in files),
        archives=archives,
        archive_bytes=sum(a.stat().st_size for a in archives),
    )


def scan_all(
    specs: Dict[str, Dict[str, Any]],
    *,
    repo_root: Path,
    manual_root: str = "datasets",
    extract: bool = True,
) -> Dict[str, LocalDataset]:
    """Scan every dataset in ``specs``. Used by ``scripts/dataset_status.py``."""
    return {
        name: scan_dataset(
            name, spec or {}, repo_root=repo_root, manual_root=manual_root, extract=extract
        )
        for name, spec in specs.items()
    }


def readiness_table(scans: Dict[str, LocalDataset], specs: Dict[str, Dict[str, Any]]):
    """A tidy DataFrame of drop-zone readiness, for notebook 01's first cell."""
    rows = []
    for name, scan in scans.items():
        spec = specs.get(name) or {}
        rows.append(
            {
                "dataset": name,
                "era": spec.get("era", ""),
                "status": "READY" if scan.available else ("EMPTY" if scan.exists else "MISSING"),
                "n_files": scan.n_files,
                "size": _human(scan.total_bytes),
                "gated": bool(spec.get("gated", False)),
                "auto_source": spec.get("primary_source", ""),
                "drop_zone": str(scan.root.name),
            }
        )
    return pd.DataFrame(rows).sort_values(["status", "dataset"]).reset_index(drop=True)


__all__ = [
    "TABULAR_SUFFIXES", "ARCHIVE_SUFFIXES",
    "canon", "resolve_column", "resolve_columns",
    "extract_archives", "list_archives", "list_data_files", "find_files",
    "iter_archive_members", "open_zip_bytes", "zip_member",
    "sniff_delimiter", "read_any", "read_many",
    "LocalDataset", "scan_dataset", "scan_all", "readiness_table",
]
