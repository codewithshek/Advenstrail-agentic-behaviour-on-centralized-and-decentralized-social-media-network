"""
aegis.text_utils
================

Text normalisation, near-duplicate detection and split hygiene for the text
branch.

Why this module exists (and why it is not just ``df.drop_duplicates()``)
-----------------------------------------------------------------------
Machine-generated-text detection is *notoriously* easy to accidentally cheat at.
Three specific traps, all of which are handled here:

1.  **Surface artefacts.** In TweepFake and HC3 the machine class carries
    tell-tale formatting (no URLs, no @mentions, straight quotes, no typos).
    A model can hit 0.97 F1 by learning "contains a t.co link => human", which
    transfers to exactly nothing. :func:`normalise_text` neutralises the
    highest-leverage artefacts by *replacing* them with stable sentinels
    instead of deleting them, so the model sees "a URL was here" without
    memorising the domain.

2.  **Length shortcuts.** HC3's ChatGPT answers are dramatically longer than the
    human ones. :func:`length_balance_report` surfaces this so you can decide
    whether to truncate, stratify or accept it — and it goes in the write-up.

3.  **Cross-split leakage.** M4 and WildJailbreak both contain templated
    near-duplicates ("Ignore previous instructions and ..."). Exact-match
    dedup misses them. :func:`dedupe_near_duplicates` uses a cheap MinHash over
    character 5-grams, which catches paraphrase-level repeats at O(n) cost, and
    :func:`assert_no_leakage` fails loudly if any survive across splits.

Everything here is dependency-light (stdlib + pandas/numpy) so it can also be
imported by the backend for inference-time preprocessing — the *same* function
must run at train and serve time or the model silently degrades.
"""

from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from .config import get_logger

log = get_logger("aegis.text")

# --------------------------------------------------------------------------- #
# Sentinels. Kept as bare uppercase words rather than special tokens so they
# survive any tokeniser without needing vocabulary surgery.
# --------------------------------------------------------------------------- #
URL_TOKEN = " HTTPURL "
MENTION_TOKEN = " @USER "
EMAIL_TOKEN = " EMAILADDR "
NUMBER_TOKEN = " NUMTOKEN "

_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_MENTION_RE = re.compile(r"(?<![\w])@[A-Za-z0-9_]{2,30}")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_HASHTAG_RE = re.compile(r"(?<![\w])#([A-Za-z0-9_]{1,60})")
_WS_RE = re.compile(r"\s+")
# Zero-width + bidi-control characters. Written as codepoint escapes on
# purpose: literal invisibles make the source unreviewable, and they are a
# known homoglyph/steganography evasion vector in 2025-era jailbreaks.
_ZERO_WIDTH_RE = re.compile(
    "[\\u200b-\\u200f\\u202a-\\u202e\\u2060-\\u2064\\u00ad\\ufeff]"
)

# Fixed permutation seed for MinHash. Changing this invalidates every cached
# signature on disk, so treat it as part of the serialisation format.
_MINHASH_SEED = 20260101

_LONG_NUM_RE = re.compile(r"\b\d[\d,._]{3,}\b")
_REPEAT_CHAR_RE = re.compile(r"(.)\1{3,}")

# Curly quotes / dashes: LLM output is systematically "typographically clean".
# Folding them removes a free win the model would otherwise take.
_PUNCT_FOLD = str.maketrans(
    {
        "‘": "'", "’": "'", "‚": "'", "‛": "'",
        "“": '"', "”": '"', "„": '"', "‟": '"',
        "′": "'", "″": '"',
        "–": "-", "—": "-", "―": "-", "−": "-",
        "…": "...",
        " ": " ", " ": " ", " ": " ",
    }
)


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
def normalise_text(
    text: object,
    *,
    lower: bool = False,
    mask_urls: bool = True,
    mask_mentions: bool = True,
    mask_emails: bool = True,
    mask_long_numbers: bool = False,
    strip_hashtag_hash: bool = False,
    fold_punctuation: bool = True,
    collapse_repeats: bool = True,
    max_chars: Optional[int] = 8000,
) -> str:
    """
    Canonicalise a single string.

    The defaults are used for corpus preparation in notebook 02. The deployed
    transformer receives raw text through ``backend/ml_service.py`` and applies
    the tokenizer configuration saved with the model artifact.

    ``lower=False`` is deliberate: casing is genuine signal. Human social posts
    have erratic capitalisation; agent output is consistently sentence-cased.
    We want the model to use that, because unlike a t.co link it is a
    *behavioural* property rather than a pipeline artefact.
    """
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    s = str(text)

    # HTML entities appear throughout the Twitter-derived corpora (&amp;, &gt;).
    s = html.unescape(s)
    # NFKC folds full-width and compatibility forms; also a cheap way to defeat
    # homoglyph-based evasion, which 2025-era jailbreak prompts use heavily.
    s = unicodedata.normalize("NFKC", s)
    s = _ZERO_WIDTH_RE.sub("", s)

    if fold_punctuation:
        s = s.translate(_PUNCT_FOLD)
    if mask_urls:
        s = _URL_RE.sub(URL_TOKEN, s)
    if mask_emails:
        s = _EMAIL_RE.sub(EMAIL_TOKEN, s)
    if mask_mentions:
        s = _MENTION_RE.sub(MENTION_TOKEN, s)
    if mask_long_numbers:
        s = _LONG_NUM_RE.sub(NUMBER_TOKEN, s)
    if strip_hashtag_hash:
        s = _HASHTAG_RE.sub(r"\1", s)
    if collapse_repeats:
        # "sooooo" -> "sooo": keeps the emphasis signal, kills the tail.
        s = _REPEAT_CHAR_RE.sub(r"\1\1\1", s)

    s = _WS_RE.sub(" ", s).strip()
    if lower:
        s = s.lower()
    if max_chars is not None and len(s) > max_chars:
        s = s[:max_chars]
    return s


def normalise_series(series: pd.Series, **kwargs) -> pd.Series:
    """Vectorised-ish wrapper (pandas ``.map`` — fine at our data scale)."""
    return series.map(lambda t: normalise_text(t, **kwargs))


def extract_hashtags(text: object) -> List[str]:
    """Lower-cased hashtags without the ``#``. Used for co-hashtag graph edges."""
    if not text:
        return []
    return [h.lower() for h in _HASHTAG_RE.findall(str(text))]


def extract_mentions(text: object) -> List[str]:
    """Lower-cased handles without the ``@``. Used for mention-graph edges."""
    if not text:
        return []
    return [m.lstrip("@").lower() for m in _MENTION_RE.findall(str(text))]


def extract_urls(text: object) -> List[str]:
    if not text:
        return []
    return _URL_RE.findall(str(text))


# --------------------------------------------------------------------------- #
# Cheap stylometry — used as interpretable side-features and for the dashboard's
# "why was this flagged" panel. Not a substitute for DeBERTa; a complement that
# an analyst can actually read.
# --------------------------------------------------------------------------- #
_FUNCTION_WORDS = (
    "the a an and or but if while of to in on at for with as that this these those "
    "is are was were be been being have has had do does did not no nor so than then"
).split()

_LLM_TELLS = (
    "as an ai", "as a language model", "i cannot", "i can't provide",
    "it's important to note", "it is important to note", "in conclusion",
    "delve into", "tapestry", "furthermore", "moreover", "additionally",
    "let's explore", "navigating the", "in today's fast-paced",
)

_INJECTION_TELLS = (
    "ignore previous instructions", "ignore prior instructions",
    "ignore all previous", "disregard the above",
    "system prompt", "you are now", "developer mode", "dan mode", "jailbreak",
    "reveal your instructions", "print your system", "bypass",
    "without any restrictions", "no ethical", "do anything now",
    "pretend you are", "roleplay as", "output the raw", "base64 decode",
    "act as an unfiltered", "execute payload",
)


def stylometric_features(text: str) -> Dict[str, float]:
    """
    ~20 interpretable features. Deliberately fast (single pass, no NLP model) so
    the backend can compute them per-request alongside the transformer score.
    """
    s = str(text or "")
    n_chars = len(s)
    words = s.split()
    n_words = len(words)
    lower = s.lower()

    if n_words == 0:
        return {k: 0.0 for k in _STYLO_KEYS}

    word_lengths = np.fromiter((len(w) for w in words), dtype=float, count=n_words)
    sentences = [x for x in re.split(r"[.!?]+", s) if x.strip()]
    unique_words = {w.lower() for w in words}
    fw_hits = sum(1 for w in words if w.lower() in _FUNCTION_WORDS)

    feats = {
        "n_chars": float(n_chars),
        "n_words": float(n_words),
        "mean_word_len": float(word_lengths.mean()),
        "std_word_len": float(word_lengths.std()),
        "n_sentences": float(len(sentences)),
        "mean_sentence_len": float(n_words / max(len(sentences), 1)),
        # LLM output has unusually *uniform* sentence length. Low variance here
        # is one of the more robust generator-agnostic signals we have.
        "sentence_len_cv": float(
            np.std([len(x.split()) for x in sentences]) /
            max(np.mean([len(x.split()) for x in sentences]), 1e-6)
        ) if sentences else 0.0,
        "type_token_ratio": float(len(unique_words) / n_words),
        "function_word_ratio": float(fw_hits / n_words),
        "punct_ratio": float(sum(c in ".,;:!?-'\"()" for c in s) / max(n_chars, 1)),
        "uppercase_ratio": float(sum(c.isupper() for c in s) / max(n_chars, 1)),
        "digit_ratio": float(sum(c.isdigit() for c in s) / max(n_chars, 1)),
        "emoji_ratio": float(
            sum(unicodedata.category(c) == "So" for c in s) / max(n_chars, 1)
        ),
        "exclaim_count": float(s.count("!")),
        "question_count": float(s.count("?")),
        "n_hashtags": float(len(_HASHTAG_RE.findall(s))),
        "n_mentions": float(len(_MENTION_RE.findall(s))),
        "n_urls": float(len(_URL_RE.findall(s))),
        "has_newline": float("\n" in s),
        # Explicit lexical detectors: not the primary classifier, but they give
        # the analyst a concrete quotable reason and they are cheap insurance.
        "llm_tell_count": float(sum(t in lower for t in _LLM_TELLS)),
        "injection_tell_count": float(sum(t in lower for t in _INJECTION_TELLS)),
        "imperative_start": float(
            bool(words) and words[0].lower() in
            {"ignore", "disregard", "forget", "pretend", "act", "output", "print", "reveal"}
        ),
    }
    return feats


_STYLO_KEYS: Tuple[str, ...] = tuple(stylometric_features("seed text here.").keys())


def stylometric_frame(texts: Iterable[str]) -> pd.DataFrame:
    """Stylometric features for a corpus, as a DataFrame with stable columns."""
    rows = [stylometric_features(t) for t in texts]
    return pd.DataFrame(rows, columns=list(_STYLO_KEYS)).fillna(0.0)


def injection_lexical_score(text: str) -> float:
    """
    Bounded [0,1] heuristic for prompt-injection / jailbreak phrasing.

    Used as (a) a weak label for unlabelled synthetic traffic and (b) a
    transparent guardrail in the backend so an obvious injection is never
    scored as benign just because the transformer is uncalibrated on it.
    """
    lower = str(text or "").lower()
    hits = sum(1 for t in _INJECTION_TELLS if t in lower)
    # Saturating rather than linear: 3 distinct tells is already conclusive.
    return float(1.0 - np.exp(-0.9 * hits))


def injection_matches(text: str) -> List[str]:
    """
    The injection tells actually present in ``text``.

    ``injection_lexical_score`` answers *how suspicious*; this answers *why*, so
    the API can quote a concrete phrase back to the analyst rather than a bare
    number.
    """
    lower = str(text or "").lower()
    return [tell for tell in _INJECTION_TELLS if tell in lower]


def injection_match_spans(text: str) -> List[Dict[str, object]]:
    """Return every lexical trigger with exact offsets into ``text``."""
    source = str(text or "")
    lower = source.lower()
    spans: List[Dict[str, object]] = []
    for tell in _INJECTION_TELLS:
        start = 0
        while True:
            at = lower.find(tell, start)
            if at < 0:
                break
            end = at + len(tell)
            spans.append(
                {
                    "start": at,
                    "end": end,
                    "text": source[at:end],
                    "trigger": tell,
                }
            )
            start = end
    return sorted(spans, key=lambda span: (int(span["start"]), int(span["end"])))


def llm_tell_matches(text: str) -> List[str]:
    """The LLM-style phrasings present in ``text``, for the same reason."""
    lower = str(text or "").lower()
    return [tell for tell in _LLM_TELLS if tell in lower]


def split_sentence_spans(text: str) -> List[Dict[str, object]]:
    """Split text while preserving each sentence's half-open source offsets."""
    source = str(text or "")
    spans: List[Dict[str, object]] = []
    for match in re.finditer(r"[^.!?\n]+(?:[.!?]+|(?=\n|$))", source):
        raw = match.group(0)
        left = len(raw) - len(raw.lstrip())
        sentence = raw.strip()
        if not sentence:
            continue
        start = match.start() + left
        spans.append(
            {
                "text": sentence,
                "start": start,
                "end": start + len(sentence),
            }
        )
    return spans


def split_sentences(text: str) -> List[str]:
    """Cheap sentence split so payload alerts can quote one sentence."""
    return [str(span["text"]) for span in split_sentence_spans(text)]


def split_posts(raw: str) -> List[str]:
    """
    Split a pasted thread into individual posts.

    Analysts paste threads either separated by blank lines or one post per line,
    so a blank-line split is tried first and single newlines are the fallback.
    """
    text = str(raw or "").replace("\r\n", "\n").strip()
    if not text:
        return []
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    if len(blocks) > 1:
        return blocks
    return [line.strip() for line in text.split("\n") if line.strip()]


# --------------------------------------------------------------------------- #
# Deduplication
# --------------------------------------------------------------------------- #
def _stable_hash(value: str) -> int:
    """Deterministic 64-bit hash. ``hash()`` is salted per-process — unusable."""
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "big")


def char_ngrams(text: str, n: int = 5) -> Set[str]:
    s = re.sub(r"\s+", " ", str(text or "").lower()).strip()
    if len(s) < n:
        return {s} if s else set()
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def minhash_signature(text: str, *, num_perm: int = 64, ngram: int = 5) -> Tuple[int, ...]:
    """
    MinHash signature over character n-grams.

    Character n-grams (not words) because paraphrased jailbreaks preserve
    character-level scaffolding even when word choice shifts. ``num_perm=64``
    gives ~±0.06 Jaccard error, which is plenty for a leakage *check*.
    """
    grams = char_ngrams(text, ngram)
    if not grams:
        return tuple([0] * num_perm)
    hashed = np.array([_stable_hash(g) for g in grams], dtype=np.uint64)
    # Universal hashing family: h_i(x) = (a_i * x + b_i) mod (2^61 - 1).
    # The permutation coefficients MUST be identical across processes and runs,
    # otherwise two invocations produce incomparable signatures — hence a fixed
    # module-level seed rather than anything derived from the input.
    rng = np.random.default_rng(_MINHASH_SEED)
    a = rng.integers(1, 2**61 - 1, size=num_perm, dtype=np.uint64)
    b = rng.integers(0, 2**61 - 1, size=num_perm, dtype=np.uint64)
    prime = np.uint64((1 << 61) - 1)
    with np.errstate(over="ignore"):
        mixed = (a[:, None] * hashed[None, :] + b[:, None]) % prime
    return tuple(int(x) for x in mixed.min(axis=1))


def exact_dedupe(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Drop byte-identical (post-normalisation) rows, keeping the first."""
    before = len(df)
    key = df[text_col].map(lambda t: _stable_hash(normalise_text(t, lower=True)))
    out = df.loc[~key.duplicated(keep="first")].copy()
    if before != len(out):
        log.info("exact_dedupe: %d -> %d rows (-%d)", before, len(out), before - len(out))
    return out


def dedupe_near_duplicates(
    df: pd.DataFrame,
    text_col: str = "text",
    *,
    threshold: float = 0.85,
    num_perm: int = 64,
    bands: int = 16,
    keep: str = "first",
) -> pd.DataFrame:
    """
    LSH-banded near-duplicate removal.

    Signatures are split into ``bands`` bands; two rows are *candidates* if any
    band matches exactly, then verified with true Jaccard on n-gram sets. This
    keeps the pairwise comparison count near-linear instead of O(n^2), which
    matters because WildJailbreak alone is 262k rows of templated text.
    """
    if df.empty:
        return df.copy()
    if num_perm % bands != 0:
        raise ValueError(f"num_perm ({num_perm}) must be divisible by bands ({bands})")

    rows_per_band = num_perm // bands
    texts = df[text_col].astype(str).tolist()
    sigs = [minhash_signature(t, num_perm=num_perm) for t in texts]

    buckets: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for idx, sig in enumerate(sigs):
        for b in range(bands):
            band = sig[b * rows_per_band : (b + 1) * rows_per_band]
            buckets[(b, _stable_hash(repr(band)))].append(idx)

    gram_cache: Dict[int, Set[str]] = {}

    def grams(i: int) -> Set[str]:
        if i not in gram_cache:
            gram_cache[i] = char_ngrams(texts[i])
        return gram_cache[i]

    drop: Set[int] = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        members = sorted(members)
        for i, left in enumerate(members):
            if left in drop:
                continue
            for right in members[i + 1 :]:
                if right in drop:
                    continue
                gl, gr = grams(left), grams(right)
                union = len(gl | gr)
                if union and len(gl & gr) / union >= threshold:
                    drop.add(right if keep == "first" else left)

    out = df.loc[~np.isin(np.arange(len(df)), sorted(drop))].copy()
    if drop:
        log.info(
            "dedupe_near_duplicates(threshold=%.2f): %d -> %d rows (-%d)",
            threshold, len(df), len(out), len(drop),
        )
    return out


def assert_no_leakage(
    splits: Dict[str, pd.DataFrame],
    text_col: str = "text",
    *,
    strict: bool = False,
) -> pd.DataFrame:
    """
    Report (and optionally reject) exact overlap between named splits.

    Called at the end of notebook 01. If this returns a non-empty frame, the
    headline metrics in notebook 02 are not trustworthy — every reported number
    in this project depends on this check passing.
    """
    hashed = {
        name: set(
            df[text_col].map(lambda t: _stable_hash(normalise_text(t, lower=True)))
        )
        for name, df in splits.items()
    }
    names = list(hashed)
    rows = []
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = hashed[left] & hashed[right]
            rows.append(
                {
                    "split_a": left,
                    "split_b": right,
                    "n_overlap": len(overlap),
                    "pct_of_smaller": 100.0
                    * len(overlap)
                    / max(min(len(hashed[left]), len(hashed[right])), 1),
                }
            )
    report = pd.DataFrame(rows)
    bad = report[report["n_overlap"] > 0] if not report.empty else report
    if not bad.empty:
        msg = f"Cross-split leakage detected:\n{bad.to_string(index=False)}"
        if strict:
            raise AssertionError(msg)
        log.error(msg)
    else:
        log.info("Leakage check passed: no exact overlap across %d splits.", len(names))
    return report


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
@dataclass
class BalanceReport:
    by_label: pd.DataFrame
    warning: Optional[str] = None

    def _repr_html_(self):  # nice rendering in Jupyter
        head = f"<p style='color:#b45309'><b>⚠ {self.warning}</b></p>" if self.warning else ""
        return head + self.by_label.to_html()


def length_balance_report(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
    *,
    ratio_alarm: float = 1.75,
) -> BalanceReport:
    """
    Check whether token length alone separates the classes.

    If the mean-length ratio between classes exceeds ``ratio_alarm``, a linear
    model on length would already score well and the transformer's reported F1
    is partly a length classifier. Notebook 02 uses this to decide whether to
    run the length-matched control experiment.
    """
    tmp = df[[text_col, label_col]].copy()
    tmp["_n_words"] = tmp[text_col].astype(str).str.split().str.len()
    tmp["_n_chars"] = tmp[text_col].astype(str).str.len()
    agg = tmp.groupby(label_col).agg(
        n=("_n_words", "size"),
        mean_words=("_n_words", "mean"),
        median_words=("_n_words", "median"),
        p95_words=("_n_words", lambda s: float(np.percentile(s, 95))),
        mean_chars=("_n_chars", "mean"),
    ).round(2)

    warning = None
    if len(agg) >= 2:
        means = agg["mean_words"].to_numpy(dtype=float)
        ratio = float(means.max() / max(means.min(), 1e-6))
        if ratio >= ratio_alarm:
            warning = (
                f"Mean length ratio between classes is {ratio:.2f}x "
                f"(>= {ratio_alarm}). Length is a confound — run the "
                f"length-matched control before quoting F1."
            )
    return BalanceReport(by_label=agg, warning=warning)


def stratified_split(
    df: pd.DataFrame,
    *,
    label_col: str = "label",
    group_col: Optional[str] = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """
    Train/val/test split that is stratified on the label and, when
    ``group_col`` is given, *grouped* so no group spans two splits.

    Grouping matters enormously here: HC3 pairs a human and a ChatGPT answer to
    the same question, and M4 pairs a human document with several generated
    continuations. Splitting those pairs across train and test leaks the topic
    and inflates F1 by several points.
    """
    from sklearn.model_selection import (
        StratifiedGroupKFold,
        StratifiedShuffleSplit,
    )

    df = df.reset_index(drop=True)
    y = df[label_col].to_numpy()

    if group_col and group_col in df.columns:
        groups = df[group_col].astype(str).to_numpy()
        n_test_splits = max(int(round(1.0 / max(test_size, 1e-6))), 2)
        sgkf = StratifiedGroupKFold(n_splits=n_test_splits, shuffle=True, random_state=seed)
        trainval_idx, test_idx = next(sgkf.split(df, y, groups))

        inner = df.iloc[trainval_idx].reset_index(drop=True)
        rel_val = val_size / max(1.0 - test_size, 1e-6)
        n_val_splits = max(int(round(1.0 / max(rel_val, 1e-6))), 2)
        sgkf2 = StratifiedGroupKFold(n_splits=n_val_splits, shuffle=True, random_state=seed)
        tr_idx, val_idx = next(
            sgkf2.split(inner, inner[label_col].to_numpy(), inner[group_col].astype(str).to_numpy())
        )
        out = {
            "train": inner.iloc[tr_idx].reset_index(drop=True),
            "validation": inner.iloc[val_idx].reset_index(drop=True),
            "test": df.iloc[test_idx].reset_index(drop=True),
        }
    else:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        trainval_idx, test_idx = next(sss.split(df, y))
        inner = df.iloc[trainval_idx].reset_index(drop=True)
        rel_val = val_size / max(1.0 - test_size, 1e-6)
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed)
        tr_idx, val_idx = next(sss2.split(inner, inner[label_col].to_numpy()))
        out = {
            "train": inner.iloc[tr_idx].reset_index(drop=True),
            "validation": inner.iloc[val_idx].reset_index(drop=True),
            "test": df.iloc[test_idx].reset_index(drop=True),
        }

    log.info(
        "stratified_split -> train=%d val=%d test=%d (grouped_by=%s)",
        len(out["train"]), len(out["validation"]), len(out["test"]), group_col,
    )
    return out


def downsample_to_balance(
    df: pd.DataFrame, label_col: str = "label", *, seed: int = 42
) -> pd.DataFrame:
    """Random-undersample the majority class to the minority count."""
    counts = df[label_col].value_counts()
    if counts.empty:
        return df.copy()
    target = int(counts.min())
    parts = [
        grp.sample(n=target, random_state=seed) if len(grp) > target else grp
        for _, grp in df.groupby(label_col, sort=False)
    ]
    out = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed)
    return out.reset_index(drop=True)


def compute_class_weights(labels: Sequence[int]) -> Dict[int, float]:
    """Inverse-frequency weights, normalised to mean 1 (keeps the LR scale sane)."""
    arr = np.asarray(list(labels))
    classes, counts = np.unique(arr, return_counts=True)
    weights = len(arr) / (len(classes) * counts)
    weights = weights / weights.mean()
    return {int(c): float(w) for c, w in zip(classes, weights)}


__all__ = [
    "URL_TOKEN", "MENTION_TOKEN",
    "normalise_text", "normalise_series",
    "extract_hashtags", "extract_mentions", "extract_urls",
    "stylometric_features", "stylometric_frame", "injection_lexical_score",
    "injection_matches", "llm_tell_matches", "split_sentences", "split_posts",
    "char_ngrams", "minhash_signature",
    "exact_dedupe", "dedupe_near_duplicates", "assert_no_leakage",
    "length_balance_report", "BalanceReport",
    "stratified_split", "downsample_to_balance", "compute_class_weights",
]
