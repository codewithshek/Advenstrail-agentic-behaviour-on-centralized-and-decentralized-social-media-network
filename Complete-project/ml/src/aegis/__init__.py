"""
AEGIS-SN — Adversarial agEntic Graph & Inference System for Social Networks.

Shared library backing the four Phase-1 notebooks. Everything the notebooks do
is importable and unit-testable from here; the notebooks themselves stay thin
so that a reviewer can read them top-to-bottom without scrolling through
200-line utility functions.

Layout
------
config            Paths, seeds, device selection, YAML config loading.
text_utils        Normalisation, near-duplicate detection, split hygiene.
dataset_loaders   One loader per dataset in the spec, each with a
                  schema-identical synthetic fallback so the pipeline is
                  runnable before any credential or access request lands.
synthetic_agents  LangChain multi-agent campaign simulator (2026 threat model).
graph_features    Interaction-graph construction + coordination physics
                  (temporal synchrony, reciprocity, burstiness, ...).
metrics           Threshold-aware classification reporting.
viz               Plot helpers used for the report figures.
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = [
    "config",
    "text_utils",
    "dataset_loaders",
    "synthetic_agents",
    "graph_features",
    "metrics",
    "viz",
]
