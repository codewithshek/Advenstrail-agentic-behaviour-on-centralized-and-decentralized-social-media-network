#!/usr/bin/env python
"""
Download the OSoMe Bot Repository datasets into ``DataSets/BotRepository/``.

    https://botometer.osome.iu.edu/bot-repository/datasets.html

Why this script exists
----------------------
That page renders nothing useful to a scraper: it is a Bootstrap shell that fetches
``datasets/index`` over AJAX and then one ``datasets/<name>/info.json`` per entry.
Reading the HTML gets you a disclaimer and a link to the Twitter developer agreement,
which is how the whole repository gets mistaken for "gated" or "empty".

It is neither. 18 of the 19 datasets are plain HTTP GETs with no request form, no
login and no signed agreement — including the FULL cresci-2017 (all five account
groups, 466 MB) and cresci-stock-2018, which is explicitly a corpus of accounts
"that act in coordinate fashion" and is therefore the closest public match to this
project's coordination target.

Usage
-----
    python scripts/fetch_bot_repository.py                 # everything except the 3 huge ones
    python scripts/fetch_bot_repository.py --all           # including caverlee/cresci-2017/2015
    python scripts/fetch_bot_repository.py --only cresci-2017 cresci-stock-2018
    python scripts/fetch_bot_repository.py --list          # show sizes, download nothing

Licensing
---------
Academic research use only, and Twitter content remains subject to the "Content
redistribution" clause of the Twitter Developer Agreement. Most archives ship user
IDs plus labels rather than tweet text for exactly that reason. Cite the paper named
in each dataset's ``publications`` field — `<dataset>/info.json` is saved alongside
the download so the citation travels with the data.
"""

from __future__ import annotations

import argparse
import json
import shutil
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

BASE = "https://botometer.osome.iu.edu/bot-repository"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEST_ROOT = REPO_ROOT.parent / "DataSets" / "BotRepository"

# Held back from the default run purely on size, not on value.
LARGE = {"caverlee-2011", "cresci-2017", "cresci-2015"}

# `kaiser` is listed in the index but its info.json points at `Astroturf.tar.gz`,
# which 404s. That is an upstream packaging bug, not a local problem.
KNOWN_BROKEN = {"kaiser"}

_CTX = ssl.create_default_context()
_HEADERS = {"User-Agent": "Mozilla/5.0 (AEGIS-SN academic research fetcher)"}


def _open(url: str, method: str = "GET"):
    request = urllib.request.Request(url, method=method, headers=_HEADERS)
    return urllib.request.urlopen(request, timeout=120, context=_CTX)


def list_datasets() -> List[str]:
    payload = _open(f"{BASE}/datasets/index").read().decode("utf-8", "replace")
    return [line.strip() for line in payload.splitlines() if line.strip()]


def fetch_info(name: str) -> Optional[Dict]:
    try:
        return json.loads(_open(f"{BASE}/datasets/{name}/info.json").read().decode("utf-8", "replace"))
    except Exception as exc:  # noqa: BLE001
        print(f"  ! {name}: no info.json ({type(exc).__name__})")
        return None


def remote_size(url: str) -> Optional[int]:
    try:
        length = _open(url, method="HEAD").headers.get("Content-Length")
        return int(length) if length and length.isdigit() else None
    except Exception:  # noqa: BLE001
        return None


def human(n: Optional[int]) -> str:
    if not n:
        return "?"
    value = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def download(name: str, info: Dict, *, force: bool = False) -> bool:
    filename = info.get("filename")
    if not filename:
        print(f"  ! {name}: info.json has no filename")
        return False

    url = f"{BASE}/datasets/{name}/{filename}"
    target_dir = DEST_ROOT / name
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / filename

    # Save the metadata regardless — it carries the citation and the licence, and
    # a dataset whose provenance you cannot state is a dataset you cannot publish on.
    (target_dir / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    expected = remote_size(url)
    if target.exists() and not force:
        actual = target.stat().st_size
        if expected is None or actual == expected:
            print(f"  = {name:<26} already present ({human(actual)})")
            return True
        print(f"  ~ {name}: size mismatch ({human(actual)} vs {human(expected)}) — refetching")

    print(f"  > {name:<26} {human(expected):>10}  {filename}")
    started = time.time()
    try:
        # Stream to a .part file so an interrupted run cannot leave a truncated
        # archive that looks complete to the next one.
        partial = target.with_suffix(target.suffix + ".part")
        with _open(url) as response, open(partial, "wb") as handle:
            shutil.copyfileobj(response, handle, length=1 << 20)
        partial.replace(target)
    except urllib.error.HTTPError as exc:
        print(f"  ! {name}: HTTP {exc.code}")
        return False
    except Exception as exc:  # noqa: BLE001
        print(f"  ! {name}: {type(exc).__name__} {exc}")
        return False

    size = target.stat().st_size
    elapsed = max(time.time() - started, 1e-6)
    print(f"    done {human(size)} in {elapsed:.1f}s ({human(int(size / elapsed))}/s)")
    return True


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="include the multi-hundred-MB archives")
    parser.add_argument("--only", nargs="*", default=None, help="specific dataset names")
    parser.add_argument("--list", action="store_true", help="list and exit")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args(argv)

    print(f"index: {BASE}/datasets/index")
    names = list_datasets()
    print(f"{len(names)} datasets listed\n")

    if args.only:
        names = [n for n in names if n in set(args.only)]
    elif not args.all:
        names = [n for n in names if n not in LARGE]
    names = [n for n in names if n not in KNOWN_BROKEN]

    if args.list:
        print(f"{'dataset':<26} {'size':>10}  description")
        print("-" * 130)
        for name in list_datasets():
            info = fetch_info(name)
            if not info:
                continue
            size = remote_size(f"{BASE}/datasets/{name}/{info.get('filename', '')}")
            note = " [large, needs --all]" if name in LARGE else ""
            print(f"{name:<26} {human(size):>10}  "
                  f"{(info.get('description') or '')[:80].replace(chr(10), ' ')}{note}")
        return 0

    DEST_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"destination: {DEST_ROOT}\n")

    ok = 0
    for name in names:
        info = fetch_info(name)
        if info and download(name, info, force=args.force):
            ok += 1

    print(f"\n{ok}/{len(names)} downloaded -> {DEST_ROOT}")
    if not args.all:
        print("Large archives skipped. Re-run with --all for "
              f"{', '.join(sorted(LARGE))} (~960 MB total).")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
