"""
Regression tests for aegis.local_store column-alias resolution.

Every case below is a real failure mode observed while feeding the resolver
deliberately-messy fixtures that mimic what the public mirrors actually ship.
The first three are the important ones: they encode bugs that were silent — the
pipeline ran, trained, and reported a good number while doing the wrong thing.

Run:  pytest ml/tests -q        (or: python ml/tests/test_local_store.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

try:
    import pytest
except ModuleNotFoundError:  # pragma: no cover
    # Minimal shim so this file is runnable via `python ml/tests/test_...py`
    # in a bare environment. Real runs should use pytest.
    import types

    class _Raises:
        def __init__(self, exc):
            self.exc = exc

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            if exc_type is None:
                raise AssertionError(f"expected {self.exc.__name__}")
            return issubclass(exc_type, self.exc)

    def _parametrize(argnames, argvalues, **_):
        def decorator(fn):
            fn._aegis_params = list(argvalues)
            return fn

        return decorator

    pytest = types.SimpleNamespace(  # type: ignore[assignment]
        mark=types.SimpleNamespace(parametrize=_parametrize),
        raises=_Raises,
    )
    sys.modules["pytest"] = pytest  # type: ignore[assignment]

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aegis.local_store import canon, resolve_column, resolve_columns  # noqa: E402


def _cols(*names):
    return pd.DataFrame(columns=list(names))


# --------------------------------------------------------------------------- #
# Silent-corruption regressions. These MUST return None rather than a wrong hit.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "columns, aliases, why",
    [
        (
            ("id", "screen_name", "followers_count"),
            ("source_id", "source", "from", "from_id"),
            "alias 'source_id' capturing a bare 'id' column builds a self-loop-only "
            "graph; the GNN then hits ~0.99 accuracy on a structureless input",
        ),
        (
            ("id", "screen_name"),
            ("target_id", "target", "to", "to_id"),
            "same failure on the destination side of an edge list",
        ),
        (
            ("id", "human_text", "machine_text", "model"),
            ("text", "content", "document"),
            "alias 'text' capturing 'human_text' silently labels every M4 row human",
        ),
    ],
)
def test_must_not_false_match(columns, aliases, why):
    assert resolve_column(_cols(*columns), aliases) is None, why


# --------------------------------------------------------------------------- #
# Matches that must keep working.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "columns, aliases, expected, why",
    [
        (("screen_name", "text", "Account_Type", "class.type"),
         ("account.type", "account_type", "label"), "Account_Type",
         "canonical form folds dots, underscores and case"),
        (("id", "followers_count", "friends_count"),
         ("followers_count", "follower_count", "followers"), "followers_count",
         "exact match wins over any substring candidate"),
        (("id", "followers", "friends"),
         ("followers_count", "follower_count", "followers"), "followers",
         "falls through the alias list to a later exact match"),
        (("id", "created", "verified"),
         ("created_at", "timestamp"), "created",
         "abbreviated header: column 'created' recovered from alias 'created_at'"),
        (("uid", "tweet_count"),
         ("statuses_count", "tweet_count", "statuses"), "tweet_count",
         "Cresci mirrors rename statuses_count -> tweet_count"),
        (("source_id", "relation", "target_id"),
         ("source_id", "source", "from"), "source_id",
         "a genuine edge file still resolves its endpoints"),
        (("source_id", "relation", "target_id"),
         ("target_id", "target", "to"), "target_id",
         "…on both sides"),
        (("Full_Text", "User"),
         ("text", "tweet", "content", "body", "full_text"), "Full_Text",
         "Twitter v2 style Full_Text column"),
        (("prompt_harm_label", "response"),
         ("prompt_harm_label", "prompt_label"), "prompt_harm_label",
         "WildGuard harm label"),
    ],
)
def test_must_match(columns, aliases, expected, why):
    assert resolve_column(_cols(*columns), aliases) == expected, why


def test_no_match_returns_none():
    assert resolve_column(_cols("a", "b"), ("text", "label")) is None


def test_required_raises():
    with pytest.raises(KeyError):
        resolve_column(_cols("a", "b"), ("text",), required=True)


def test_canon():
    assert canon(" Account.Type ") == "accounttype"
    assert canon("public_metrics.followers_count") == "publicmetricsfollowerscount"


def test_resolve_columns_table():
    df = _cols("id", "text", "Account_Type", "class.type")
    got = resolve_columns(df, {
        "text": ["text", "tweet"],
        "label": ["account.type", "label"],
        "generator": ["class_type", "generator"],
        "missing": ["nope_at_all"],
    })
    assert got == {
        "text": "text",
        "label": "Account_Type",
        "generator": "class.type",
        "missing": None,
    }


if __name__ == "__main__":  # allow running without pytest installed
    import traceback

    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        argsets = getattr(fn, "_aegis_params", None)
        if argsets is None:
            argsets = []
            for mark in getattr(fn, "pytestmark", []):
                if getattr(mark, "name", "") == "parametrize":
                    argsets = list(mark.args[1])
        try:
            if argsets:
                for args in argsets:
                    fn(*args)
                    passed += 1
            else:
                fn()
                passed += 1
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
