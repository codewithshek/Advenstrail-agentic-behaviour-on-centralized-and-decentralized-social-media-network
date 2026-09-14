#!/usr/bin/env python
"""
Build ``ml/notebooks/*.ipynb`` from the percent-format sources in
``ml/notebooks/_src/*.py``.

Why a build step instead of just committing .ipynb files
--------------------------------------------------------
A committed notebook is a JSON blob with embedded outputs and execution counts.
Every run rewrites it, so ``git diff`` is unreadable, merges conflict on cells
nobody touched, and review comments land on line numbers that mean nothing. The
percent format (``# %%`` / ``# %% [markdown]``, the same convention jupytext and
VS Code use) keeps the source as plain Python: reviewable, greppable, lintable,
and importable by pytest.

The generated notebooks are the artefact you open in Jupyter. The ``_src/``
files are the thing you edit and commit.

Usage
-----
    python scripts/build_notebooks.py              # build all
    python scripts/build_notebooks.py 02 03        # build the ones matching
    python scripts/build_notebooks.py --check      # fail if a notebook is stale

``--check`` is the CI hook: it rebuilds in memory and compares, so a PR that
edits a notebook without regenerating it gets caught.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "ml" / "notebooks" / "_src"
OUT_DIR = REPO_ROOT / "ml" / "notebooks"

CELL_MARKER = "# %%"
MARKDOWN_MARKER = "# %% [markdown]"

KERNELSPEC = {
    "display_name": "Python 3 (AEGIS-SN)",
    "language": "python",
    "name": "python3",
}


def _split_cells(text: str) -> list[tuple[str, str]]:
    """Split percent-format source into ``(cell_type, source)`` pairs."""
    lines = text.splitlines()
    cells: list[tuple[str, list[str]]] = []
    current_type = "code"
    current: list[str] = []

    for line in lines:
        stripped = line.rstrip()
        if stripped.startswith(CELL_MARKER):
            if current:
                cells.append((current_type, current))
            current = []
            current_type = "markdown" if stripped.startswith(MARKDOWN_MARKER) else "code"
            continue
        current.append(line)
    if current:
        cells.append((current_type, current))

    out: list[tuple[str, str]] = []
    for cell_type, body in cells:
        if cell_type == "markdown":
            # Markdown cells are written as comments in the .py so the file
            # stays valid Python; strip the leading "# " here.
            cleaned = []
            for line in body:
                if line.startswith("# "):
                    cleaned.append(line[2:])
                elif line.strip() == "#":
                    cleaned.append("")
                else:
                    cleaned.append(line)
            body = cleaned
        source = "\n".join(body).strip("\n")
        if not source.strip():
            continue
        out.append((cell_type, source))
    return out


def _to_notebook(cells: list[tuple[str, str]]) -> dict:
    payload: list[dict] = []
    for index, (cell_type, source) in enumerate(cells):
        # nbformat wants a list of lines with the newline retained on all but
        # the last. Writing a single string "works" in Jupyter but breaks
        # nbformat.validate and confuses some diff tools.
        lines = source.splitlines(keepends=True)
        # Notebook format 4.5 requires stable cell ids. Older nbformat versions
        # silently inserted them at execution time; current versions warn and
        # future versions will reject id-less cells.
        cell_id = hashlib.sha1(
            f"{index}:{cell_type}:{source}".encode()
        ).hexdigest()[:12]
        cell: dict = {
            "cell_type": cell_type,
            "id": cell_id,
            "metadata": {},
            "source": lines,
        }
        if cell_type == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        payload.append(cell)

    return {
        "cells": payload,
        "metadata": {
            "kernelspec": KERNELSPEC,
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build(source: Path, *, check: bool) -> bool:
    """Build one notebook. Returns True when the on-disk file is up to date."""
    target = OUT_DIR / (source.stem + ".ipynb")
    notebook = _to_notebook(_split_cells(source.read_text(encoding="utf-8")))
    rendered = json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"

    if check:
        if not target.exists():
            print(f"MISSING  {target.relative_to(REPO_ROOT)}")
            return False
        # Compare cell sources only. Executing a notebook rewrites outputs and
        # execution_count, and that must not be treated as "stale".
        existing = json.loads(target.read_text(encoding="utf-8"))
        same = [c["source"] for c in existing.get("cells", [])] == [
            c["source"] for c in notebook["cells"]
        ]
        print(("OK       " if same else "STALE    ") + str(target.relative_to(REPO_ROOT)))
        return same

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(rendered, encoding="utf-8")
    n_code = sum(1 for c in notebook["cells"] if c["cell_type"] == "code")
    n_md = len(notebook["cells"]) - n_code
    print(f"wrote {target.relative_to(REPO_ROOT)}  ({n_code} code, {n_md} markdown)")
    return True


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filters", nargs="*", help="substring match on the notebook name")
    parser.add_argument("--check", action="store_true", help="verify without writing")
    args = parser.parse_args(argv)

    if not SRC_DIR.exists():
        print(f"no source directory at {SRC_DIR}", file=sys.stderr)
        return 1

    sources = sorted(SRC_DIR.glob("*.py"))
    if args.filters:
        sources = [p for p in sources if any(f in p.name for f in args.filters)]
    if not sources:
        print("nothing to build", file=sys.stderr)
        return 1

    ok = all(build(path, check=args.check) for path in sources)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
