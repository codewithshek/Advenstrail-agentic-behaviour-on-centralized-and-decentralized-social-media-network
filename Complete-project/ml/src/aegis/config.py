"""
aegis.config
============

Repo-root discovery, path management, seeding, device selection and YAML config
loading.

Design notes
------------
*   **Root discovery walks upward for a marker file.** Notebooks are executed
    with `cwd` set to wherever the user happened to launch Jupyter — the repo
    root, `ml/`, or `ml/notebooks/`. Hard-coding `../..` breaks the moment
    someone runs a cell from a different place, and that failure is confusing.
    We instead walk up looking for `requirements.txt` + `ml/`.
*   **Env vars beat the YAML file.** CI and the smoke-test harness need to flip
    `AEGIS_SMOKE_TEST` and `AEGIS_DEVICE` without editing tracked files.
*   **Seeding is centralised and covers Python/NumPy/torch.** Reproducibility
    is not a nice-to-have for a security paper — a reviewer must be able to
    regenerate the exact confusion matrix.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

# --------------------------------------------------------------------------- #
# Optional imports — the module must be importable in a bare environment so
# that `python -c "import aegis.config"` works before the heavy deps land.
# --------------------------------------------------------------------------- #
try:  # pragma: no cover
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:  # pragma: no cover
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

try:  # pragma: no cover
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore[assignment]


LOGGER_NAME = "aegis"
_ROOT_MARKERS = ("requirements.txt", "ml")
_TRUTHY = {"1", "true", "yes", "y", "on"}


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def _stdout_supports_unicode() -> bool:
    """
    Can we safely emit box-drawing characters to stdout?

    On Windows a bare `python script.py` gets a cp1252 stdout, and logging a
    '│' raises UnicodeEncodeError *inside the handler*. logging swallows that
    into a "--- Logging error ---" dump on stderr, so every INFO line becomes a
    20-line traceback and the actual run output is unreadable. Jupyter and most
    Linux/macOS terminals are UTF-8 and are fine.

    We try to upgrade the stream to UTF-8 first (Python 3.7+ allows this) and
    only fall back to an ASCII format if that is refused.
    """
    stream = getattr(sys, "stdout", None)
    encoding = (getattr(stream, "encoding", None) or "").lower()
    if "utf" in encoding:
        return True
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(encoding="utf-8", errors="replace")
            return True
        except Exception:  # noqa: BLE001 — pytest/IPython capture objects refuse
            pass
    return False


def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    """Return a configured logger, idempotently (safe to call from any cell)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        separator = "\u2502" if _stdout_supports_unicode() else "|"
        handler.setFormatter(
            logging.Formatter(
                fmt=f"%(asctime)s {separator} %(levelname)-7s {separator} %(name)s {separator} %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(os.environ.get("AEGIS_LOG_LEVEL", "INFO").upper())
        logger.propagate = False
    return logger


log = get_logger()


#: Glyphs used in log output, degraded to ASCII when stdout cannot encode them.
#: Import these instead of hard-coding '✔'/'─' anywhere that gets logged.
GLYPHS: Dict[str, str] = (
    {"ok": "\u2714", "gen": "\u2699", "warn": "\u26a0", "rule": "\u2500"}
    if _stdout_supports_unicode()
    else {"ok": "[OK]", "gen": "[GEN]", "warn": "[!]", "rule": "-"}
)


# --------------------------------------------------------------------------- #
# Root discovery
# --------------------------------------------------------------------------- #
def find_repo_root(start: Optional[Path] = None) -> Path:
    """
    Walk upward from ``start`` (default: cwd, then this file) until a directory
    containing every entry in ``_ROOT_MARKERS`` is found.

    Falls back to two levels above this file, which is correct for the shipped
    layout (``ml/src/aegis/config.py`` -> repo root is three parents up).
    """
    candidates = []
    if start is not None:
        candidates.append(Path(start).resolve())
    candidates.append(Path.cwd().resolve())
    candidates.append(Path(__file__).resolve())

    for candidate in candidates:
        for parent in [candidate, *candidate.parents]:
            if all((parent / marker).exists() for marker in _ROOT_MARKERS):
                return parent

    # Deterministic fallback: ml/src/aegis/config.py -> ../../../
    fallback = Path(__file__).resolve().parents[3]
    log.warning("Repo root markers not found; falling back to %s", fallback)
    return fallback


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Paths:
    """Every filesystem location the pipeline touches, resolved absolutely."""

    root: Path

    # --- data lake tiers ---------------------------------------------------
    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def raw(self) -> Path:
        """Byte-for-byte downloads. Treated as immutable."""
        return self.data / "raw"

    @property
    def interim(self) -> Path:
        """Per-dataset harmonised frames (one parquet per source)."""
        return self.data / "interim"

    @property
    def processed(self) -> Path:
        """Model-ready splits. This is what notebooks 02-04 consume."""
        return self.data / "processed"

    @property
    def external(self) -> Path:
        """Third-party caches (HF hub, Kaggle zips)."""
        return self.data / "external"

    @property
    def synthetic(self) -> Path:
        """Generated 2026 agentic-campaign corpora."""
        return self.data / "synthetic"

    # --- artefacts ---------------------------------------------------------
    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def text_model(self) -> Path:
        return self.models / "text_model"

    @property
    def graph_model(self) -> Path:
        return self.models / "graph_model"

    @property
    def fusion_model(self) -> Path:
        return self.models / "fusion_model"

    # --- reporting ---------------------------------------------------------
    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def figures(self) -> Path:
        return self.reports / "figures"

    # --- code --------------------------------------------------------------
    @property
    def ml(self) -> Path:
        return self.root / "ml"

    @property
    def src(self) -> Path:
        return self.ml / "src"

    @property
    def notebooks(self) -> Path:
        return self.ml / "notebooks"

    @property
    def configs(self) -> Path:
        return self.ml / "configs"

    @property
    def manifest(self) -> Path:
        """Provenance ledger: which datasets are REAL vs SYNTHETIC_FALLBACK."""
        return self.data / "manifest.json"

    def ensure(self) -> "Paths":
        """Create every writable directory. Idempotent."""
        for p in (
            self.raw,
            self.interim,
            self.processed,
            self.external,
            self.synthetic,
            self.text_model,
            self.graph_model,
            self.fusion_model,
            self.figures,
        ):
            p.mkdir(parents=True, exist_ok=True)
        return self

    def describe(self) -> str:
        return "\n".join(
            f"  {name:<16} {getattr(self, name)}"
            for name in (
                "root",
                "raw",
                "interim",
                "processed",
                "synthetic",
                "models",
                "figures",
            )
        )


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
@dataclass
class Settings:
    """Merged view of ``ml/configs/default.yaml`` + environment overrides."""

    paths: Paths
    raw: Dict[str, Any] = field(default_factory=dict)
    seed: int = 42
    smoke_test: bool = True
    device: str = "auto"

    # ---- dict-ish access so notebooks can do cfg["text_model"]["epochs"] ---
    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def section(self, *keys: str, default: Any = None) -> Any:
        """Safe nested lookup: ``cfg.section("text_model", "epochs")``."""
        node: Any = self.raw
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    # ---- convenience ------------------------------------------------------
    @property
    def text_model(self) -> Dict[str, Any]:
        return self.raw.get("text_model", {})

    @property
    def graph_model(self) -> Dict[str, Any]:
        return self.raw.get("graph_model", {})

    @property
    def fusion_model(self) -> Dict[str, Any]:
        return self.raw.get("fusion_model", {})

    @property
    def text_datasets(self) -> Dict[str, Any]:
        return self.raw.get("text_datasets", {})

    @property
    def graph_datasets(self) -> Dict[str, Any]:
        return self.raw.get("graph_datasets", {})

    @property
    def synthetic(self) -> Dict[str, Any]:
        return self.raw.get("synthetic", {})

    @property
    def row_cap(self) -> Optional[int]:
        """Rows to keep per dataset when smoke-testing (``None`` = no cap)."""
        if not self.smoke_test:
            return None
        return int(
            self.section("project", "smoke_test_rows_per_dataset", default=1500)
        )


_DEFAULT_CONFIG: Dict[str, Any] = {
    "project": {"seed": 42, "smoke_test": True, "smoke_test_rows_per_dataset": 1500},
    "text_model": {
        "base_checkpoint": "microsoft/deberta-v3-base",
        "max_length": 256,
        "num_labels": 2,
        "epochs": 3,
        "batch_size": 16,
        "eval_batch_size": 64,
        "learning_rate": 2e-5,
    },
    "graph_model": {"architecture": "sage", "hidden_channels": 128, "epochs": 200},
    "fusion_model": {"strategy": "stacking", "meta_learner": "logistic"},
}


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in _TRUTHY


def load_config(
    config_path: Optional[Path] = None,
    *,
    ensure_dirs: bool = True,
    load_env: bool = True,
    quiet: bool = False,
) -> Settings:
    """
    Load ``ml/configs/default.yaml`` and apply environment overrides.

    Parameters
    ----------
    config_path
        Explicit YAML path. Defaults to ``<root>/ml/configs/default.yaml``.
    ensure_dirs
        Create the data/model/report directories.
    load_env
        Read ``<root>/.env`` via python-dotenv if available.
    """
    root = find_repo_root()
    paths = Paths(root=root)

    if load_env and load_dotenv is not None:
        env_file = root / ".env"
        if env_file.exists():
            load_dotenv(env_file, override=False)

    cfg_path = Path(config_path) if config_path else paths.configs / "default.yaml"
    raw: Dict[str, Any]
    if yaml is not None and cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    else:
        if yaml is None:
            log.warning("PyYAML unavailable — using built-in defaults.")
        else:
            log.warning("Config not found at %s — using built-in defaults.", cfg_path)
        raw = json.loads(json.dumps(_DEFAULT_CONFIG))  # deep copy

    # ---- environment overrides -------------------------------------------
    seed = int(os.environ.get("AEGIS_SEED", raw.get("project", {}).get("seed", 42)))
    smoke = _env_bool(
        "AEGIS_SMOKE_TEST", bool(raw.get("project", {}).get("smoke_test", True))
    )
    device = os.environ.get(
        "AEGIS_DEVICE", raw.get("project", {}).get("device", "auto")
    )

    raw.setdefault("project", {})
    raw["project"].update({"seed": seed, "smoke_test": smoke, "device": device})

    # Synthetic backend override (offline | openai | anthropic).
    synth_backend = os.environ.get("AEGIS_SYNTH_BACKEND")
    if synth_backend:
        raw.setdefault("synthetic", {})["backend"] = synth_backend.strip().lower()

    settings = Settings(
        paths=paths, raw=raw, seed=seed, smoke_test=smoke, device=device
    )

    if ensure_dirs:
        paths.ensure()

    # Make `import aegis` work from any notebook without a pip install -e.
    src = str(paths.src)
    if src not in sys.path:
        sys.path.insert(0, src)

    if not quiet:
        log.info("AEGIS-SN config loaded from %s", cfg_path)
        log.info("root=%s | seed=%d | smoke_test=%s | device=%s",
                 paths.root, seed, smoke, resolve_device(device))
        if smoke:
            log.warning(
                "SMOKE_TEST is ON: datasets capped at %s rows and epochs reduced. "
                "Set AEGIS_SMOKE_TEST=0 for a publication run.",
                settings.row_cap,
            )
    return settings


# --------------------------------------------------------------------------- #
# Seeding & device
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 42, *, deterministic: bool = True) -> int:
    """
    Seed Python, NumPy and torch (CPU + all CUDA devices).

    ``deterministic=True`` also pins cuDNN, which costs a little throughput but
    is what makes a reported F1 reproducible on the same hardware.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    try:  # torch is optional at import time
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:  # pragma: no cover
        pass
    return seed


def resolve_device(preference: str = "auto") -> str:
    """Map ``auto`` to the best available backend; validate explicit choices."""
    preference = (preference or "auto").lower()
    try:
        import torch
    except Exception:  # pragma: no cover
        return "cpu"

    if preference == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if preference == "cuda" and not torch.cuda.is_available():
        log.warning("device=cuda requested but CUDA is unavailable — using CPU.")
        return "cpu"
    if preference == "mps" and not (
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    ):
        log.warning("device=mps requested but MPS is unavailable — using CPU.")
        return "cpu"
    return preference


def torch_device(preference: str = "auto"):
    """Return an actual ``torch.device``."""
    import torch

    return torch.device(resolve_device(preference))


# --------------------------------------------------------------------------- #
# Provenance manifest
# --------------------------------------------------------------------------- #
def read_manifest(paths: Paths) -> Dict[str, Any]:
    if paths.manifest.exists():
        try:
            return json.loads(paths.manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log.warning("manifest.json is corrupt — starting a fresh one.")
    return {"schema_version": 1, "datasets": {}}


def write_manifest_entry(paths: Paths, name: str, entry: Dict[str, Any]) -> None:
    """
    Record how a dataset was obtained.

    This is the single most important piece of experimental hygiene in the
    project: it makes it impossible to accidentally report a metric computed on
    a synthetic fallback as if it came from the real corpus.
    """
    manifest = read_manifest(paths)
    manifest["datasets"][name] = entry
    paths.manifest.parent.mkdir(parents=True, exist_ok=True)
    paths.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )


def manifest_summary(paths: Paths):
    """Return a tidy DataFrame (or list of dicts) of dataset provenance."""
    manifest = read_manifest(paths)
    rows = []
    for name, entry in sorted(manifest.get("datasets", {}).items()):
        rows.append(
            {
                "dataset": name,
                "provenance": entry.get("provenance"),
                "source": entry.get("source"),
                "rows": entry.get("rows"),
                "sha256_16": (entry.get("sha256") or "")[:16],
                "retrieved_at": entry.get("retrieved_at"),
                "note": (entry.get("note") or "")[:80],
            }
        )
    try:
        import pandas as pd

        return pd.DataFrame(rows)
    except Exception:  # pragma: no cover
        return rows


def has_credentials(kind: str) -> bool:
    """Cheap capability probe used by the loaders to pick a strategy."""
    kind = kind.lower()
    if kind in {"hf", "huggingface"}:
        return bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    if kind == "kaggle":
        if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
            return True
        return (Path.home() / ".kaggle" / "kaggle.json").exists()
    if kind == "openai":
        return bool(os.environ.get("OPENAI_API_KEY"))
    if kind == "anthropic":
        return bool(os.environ.get("ANTHROPIC_API_KEY"))
    return False


__all__ = [
    "Paths",
    "Settings",
    "find_repo_root",
    "load_config",
    "set_seed",
    "resolve_device",
    "torch_device",
    "get_logger",
    "read_manifest",
    "write_manifest_entry",
    "manifest_summary",
    "has_credentials",
]
