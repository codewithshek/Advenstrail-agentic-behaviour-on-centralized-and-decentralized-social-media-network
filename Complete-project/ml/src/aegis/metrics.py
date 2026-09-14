"""
aegis.metrics
=============

Threshold-aware classification reporting. Every number that ends up in the
write-up is computed here.

Why a module instead of three cells
-----------------------------------
Notebooks 02 (text), 03 (graph) and 04 (fusion) all report "F1". If each one
calls ``sklearn.metrics.f1_score`` with its own defaults, they are silently
reporting three different quantities — macro vs binary averaging, 0.5 vs a
tuned threshold, ``zero_division`` raising vs returning 0. That discrepancy is
invisible in a notebook and fatal in a results table, because the fusion
ablation in notebook 04 is a *comparison* between those three numbers. So the
comparison is only meaningful if one function produced all of them.

Three opinions baked in
-----------------------
1.  **The threshold is part of the result.** A classifier does not have an F1;
    a classifier *at an operating point* has an F1. :class:`ClassificationReport`
    therefore carries the threshold it was computed at, and
    :func:`tune_threshold` exists so that the operating point is chosen
    explicitly and reported, rather than inherited from sklearn's 0.5.

2.  **F-beta with beta=1.5, not F1.** See :func:`tune_threshold`. The value
    lives in ``fusion_model.fbeta`` in the YAML so the notebooks and the
    backend cannot drift apart.

3.  **A point estimate is not a result.** With ``smoke_test: true`` the splits
    are ~1500 rows, of which the positive class may be a few hundred. The
    sampling noise on F1 at that size is several points — larger than most of
    the ablation deltas we want to claim. :func:`bootstrap_ci` is therefore not
    optional garnish; a delta whose intervals overlap is not a finding.

Degenerate inputs are the norm here, not the exception: per-generator slices in
notebook 02 and per-relation slices in notebook 03 routinely contain a single
class or a handful of rows. Nothing in this module raises on those — it returns
NaN, logs why, and (in :func:`per_group_report`) flags the row, because a
silent exception halfway through a 40-group loop costs more than a NaN.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .config import Settings, get_logger

log = get_logger("aegis.metrics")

#: Cost-sensitive default, mirroring ``fusion_model.fbeta``. Read the rationale
#: in :func:`tune_threshold` before changing it.
DEFAULT_BETA: float = 1.5

#: Analyst-facing triage bands, mirroring ``fusion_model.decision_thresholds``.
DEFAULT_BANDS: Dict[str, float] = {"monitor": 0.35, "investigate": 0.60, "escalate": 0.82}

#: Band names in increasing severity. Used as the Categorical ordering so that
#: ``bands.max()`` and ``sort_values()`` mean what an analyst expects.
BAND_ORDER: Tuple[str, ...] = ("clear", "monitor", "investigate", "escalate")

#: Metric names accepted by :func:`tune_threshold` and :func:`bootstrap_ci`.
METRIC_NAMES: Tuple[str, ...] = (
    "f1", "fbeta", "precision", "recall", "accuracy", "mcc", "roc_auc", "pr_auc", "youden",
)

_ScoreKind = Literal["binary", "probability", "score"]

# sklearn costs ~1s to import and pulls in scipy. `import aegis.metrics` happens
# at the top of every notebook, often just to get DEFAULT_BANDS, so the import
# is deferred to first actual use and then cached.
_SKM: Any = None


def _sk() -> Any:
    """Lazily import and cache ``sklearn.metrics``."""
    global _SKM
    if _SKM is None:
        from sklearn import metrics as skm

        _SKM = skm
    return _SKM


# --------------------------------------------------------------------------- #
# Input hygiene
# --------------------------------------------------------------------------- #
def _prepare(
    y_true: Sequence[Any], y_score: Sequence[Any]
) -> Tuple[np.ndarray, np.ndarray, _ScoreKind]:
    """
    Coerce ``(y_true, y_score)`` into aligned float arrays and classify the
    score scale.

    Three shapes of input reach this project and all three are legitimate:
    calibrated probabilities from the fusion meta-learner, hard 0/1 predictions
    from a rules baseline, and unbounded decision-function values from an
    uncalibrated SVM/logit. They are distinguished here rather than by asking
    the caller, because the caller is a notebook cell that will get it wrong.

    Missing scores are imputed to 0.0, not dropped. A row the pipeline failed
    to score is a row an analyst never sees, so it must count as a miss —
    dropping it would quietly improve recall by deleting the evidence of the
    failure.
    """
    yt = pd.to_numeric(pd.Series(np.asarray(y_true).ravel()), errors="coerce").to_numpy(dtype=float)
    ys = pd.to_numeric(pd.Series(np.asarray(y_score).ravel()), errors="coerce").to_numpy(dtype=float)

    if yt.shape[0] != ys.shape[0]:
        raise ValueError(f"y_true and y_score have different lengths: {yt.shape[0]} vs {ys.shape[0]}")

    usable = ~np.isnan(yt)
    if not usable.all():
        log.warning("dropping %d row(s) with a missing/non-numeric label", int((~usable).sum()))
        yt, ys = yt[usable], ys[usable]

    missing_scores = np.isnan(ys)
    if missing_scores.any():
        log.warning(
            "%d score(s) were NaN/None and are counted as 0.0 (i.e. as misses). "
            "That is deliberate — an unscored row is an undetected row.",
            int(missing_scores.sum()),
        )
        ys = np.where(missing_scores, 0.0, ys)

    yt = (yt > 0.5).astype(int)

    finite = ys[np.isfinite(ys)]
    if finite.size and np.isin(finite, (0.0, 1.0)).all():
        kind: _ScoreKind = "binary"
    elif finite.size == 0 or (finite.min() >= 0.0 and finite.max() <= 1.0):
        kind = "probability"
    else:
        kind = "score"
        log.warning(
            "scores fall outside [0, 1] (min=%.4g max=%.4g) — treating them as an "
            "uncalibrated decision function. Ranking metrics are still valid; the "
            "Brier score is not and is reported as NaN.",
            float(finite.min()), float(finite.max()),
        )
    return yt, ys, kind


def _fmt(value: float, digits: int = 3) -> str:
    return "  nan" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


# --------------------------------------------------------------------------- #
# The report object
# --------------------------------------------------------------------------- #
@dataclass(repr=False)
class ClassificationReport:
    """
    One evaluation of one model at one operating point.

    Deliberately a flat record of floats rather than a nested dict: it has to
    survive :func:`aegis.io_utils.save_json`, a pandas concat in
    :func:`compare_reports`, and being eyeballed in a log line, and a flat
    record is the only shape that does all three well.
    """

    accuracy: float
    precision: float
    recall: float
    f1: float
    fbeta: float
    roc_auc: float
    pr_auc: float
    mcc: float
    brier: float
    tn: int
    fp: int
    fn: int
    tp: int
    support: int
    threshold: float
    n_positive: int
    beta: float = DEFAULT_BETA
    name: str = "report"

    # ---- derived, cheap enough to be properties --------------------------- #
    @property
    def positive_rate(self) -> float:
        """Base rate of the positive class. Context for every other number."""
        return self.n_positive / self.support if self.support else float("nan")

    @property
    def alert_rate(self) -> float:
        """
        Fraction of the population that would land in an analyst's queue.

        Reported alongside recall because the pair is what a deployment
        decision actually turns on: 0.95 recall at a 40% alert rate is not a
        detector, it is a coin flip with extra steps.
        """
        flagged = self.tp + self.fp
        return flagged / self.support if self.support else float("nan")

    @property
    def specificity(self) -> float:
        negatives = self.tn + self.fp
        return self.tn / negatives if negatives else float("nan")

    # ---- serialisation ---------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = asdict(self)
        out.update(
            {
                "positive_rate": self.positive_rate,
                "alert_rate": self.alert_rate,
                "specificity": self.specificity,
            }
        )
        return out

    def to_frame(self) -> pd.DataFrame:
        """Single-row frame indexed by ``name`` — concat-ready for ablations."""
        data = self.to_dict()
        label = data.pop("name")
        return pd.DataFrame([data], index=pd.Index([label], name="model"))

    def summary(self) -> str:
        """Multi-line human summary for a log line or a notebook print."""
        return (
            f"{self.name} @ threshold={_fmt(self.threshold, 2)}\n"
            f"  support={self.support} positives={self.n_positive} "
            f"({self.positive_rate:.1%}) alerts={self.alert_rate:.1%}\n"
            f"  precision={_fmt(self.precision)} recall={_fmt(self.recall)} "
            f"f1={_fmt(self.f1)} f{self.beta:g}={_fmt(self.fbeta)}\n"
            f"  roc_auc={_fmt(self.roc_auc)} pr_auc={_fmt(self.pr_auc)} "
            f"mcc={_fmt(self.mcc)} brier={_fmt(self.brier)}\n"
            f"  confusion  tn={self.tn} fp={self.fp} fn={self.fn} tp={self.tp}"
        )

    def __repr__(self) -> str:
        return (
            f"<{self.name} P={_fmt(self.precision)} R={_fmt(self.recall)} "
            f"F1={_fmt(self.f1)} F{self.beta:g}={_fmt(self.fbeta)} "
            f"AUC={_fmt(self.roc_auc)} AP={_fmt(self.pr_auc)} "
            f"@thr={_fmt(self.threshold, 2)} n={self.support}>"
        )


# --------------------------------------------------------------------------- #
# Core evaluation
# --------------------------------------------------------------------------- #
def evaluate(
    y_true: Sequence[Any],
    y_score: Sequence[Any],
    *,
    threshold: float = 0.5,
    beta: float = DEFAULT_BETA,
    name: str = "report",
) -> ClassificationReport:
    """
    Full metric set for one operating point.

    ``y_score`` may be probabilities, an uncalibrated decision function, or hard
    0/1 predictions; the scale is detected. For hard predictions the ranking
    metrics (ROC-AUC, PR-AUC) and the Brier score are returned as NaN rather
    than computed: ROC-AUC on two distinct values is mathematically defined but
    it equals balanced accuracy, and printing that in the same column as a real
    AUC in :func:`compare_reports` invites a false comparison.

    A single-class ``y_true`` also yields NaN for ROC-AUC/PR-AUC with a warning
    instead of an exception. This is not a hypothetical: the ``cohere`` and
    ``flant5`` holdout slices in notebook 02 and several relation slices in
    notebook 03 contain positives only.
    """
    skm = _sk()
    yt, ys, kind = _prepare(y_true, y_score)

    support = int(yt.shape[0])
    n_positive = int(yt.sum())
    if support == 0:
        log.warning("evaluate(%s): empty input after cleaning — returning an all-NaN report", name)
        nan = float("nan")
        return ClassificationReport(
            accuracy=nan, precision=nan, recall=nan, f1=nan, fbeta=nan, roc_auc=nan,
            pr_auc=nan, mcc=nan, brier=nan, tn=0, fp=0, fn=0, tp=0, support=0,
            threshold=float(threshold), n_positive=0, beta=float(beta), name=name,
        )

    y_pred = ys.astype(int) if kind == "binary" else (ys >= threshold).astype(int)

    # labels=[0,1] forces a 2x2 even when a slice is single-class; without it
    # sklearn returns a 1x1 and the ravel() below raises.
    tn, fp, fn, tp = skm.confusion_matrix(yt, y_pred, labels=[0, 1]).ravel()

    single_class = n_positive in (0, support)
    if single_class:
        log.warning(
            "evaluate(%s): y_true has a single class (%d positives / %d rows). "
            "ROC-AUC and PR-AUC are undefined and reported as NaN.",
            name, n_positive, support,
        )

    if single_class or kind == "binary":
        roc_auc = pr_auc = float("nan")
    else:
        roc_auc = float(skm.roc_auc_score(yt, ys))
        pr_auc = float(skm.average_precision_score(yt, ys))

    if kind == "probability":
        brier = float(skm.brier_score_loss(yt, np.clip(ys, 0.0, 1.0)))
    else:
        brier = float("nan")

    return ClassificationReport(
        accuracy=float(skm.accuracy_score(yt, y_pred)),
        precision=float(skm.precision_score(yt, y_pred, zero_division=0)),
        recall=float(skm.recall_score(yt, y_pred, zero_division=0)),
        f1=float(skm.f1_score(yt, y_pred, zero_division=0)),
        fbeta=float(skm.fbeta_score(yt, y_pred, beta=beta, zero_division=0)),
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        mcc=float(skm.matthews_corrcoef(yt, y_pred)) if not single_class else float("nan"),
        brier=brier,
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
        support=support,
        threshold=float("nan") if kind == "binary" else float(threshold),
        n_positive=n_positive,
        beta=float(beta),
        name=name,
    )


# --------------------------------------------------------------------------- #
# Operating point selection
# --------------------------------------------------------------------------- #
def _candidate_thresholds(scores: np.ndarray, *, n_grid: int) -> np.ndarray:
    """
    Threshold grid: score quantiles, not a uniform 0..1 linspace.

    Fusion scores are heavily piled up near 0 (most accounts are obviously
    benign), so a uniform grid spends 80% of its candidates in a region where
    nothing changes and resolves the interesting 0.6-0.95 range with a handful
    of points. Quantiles put the candidates where the data actually is. 0.5 and
    the configured band edges are appended so the tuned point can always be
    compared against the defaults on the same grid.
    """
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return np.array([0.5])
    quantiles = np.quantile(finite, np.linspace(0.0, 1.0, max(int(n_grid), 3)))
    extras = [0.5, *DEFAULT_BANDS.values()]
    grid = np.unique(np.concatenate([quantiles, np.asarray(extras, dtype=float)]))
    # Sit strictly above each observed value so a threshold never lands exactly
    # on a mass of tied scores, where `>=` flips a whole block at once.
    return np.unique(np.clip(grid, finite.min() - 1e-9, finite.max() + 1e-9))


def threshold_sweep(
    y_true: Sequence[Any],
    y_score: Sequence[Any],
    *,
    beta: float = DEFAULT_BETA,
    n_grid: int = 201,
) -> pd.DataFrame:
    """
    Precision / recall / F1 / F-beta / alert-rate across the threshold grid.

    Returned as a frame rather than plotted here so that
    :func:`aegis.viz.plot_threshold_sweep` and any tuning code share one
    implementation — the curve in the report and the operating point in the
    model card must come from the same arithmetic.
    """
    yt, ys, kind = _prepare(y_true, y_score)
    if kind == "binary":
        log.warning("threshold_sweep: scores are hard 0/1 labels, so the sweep is a step function")

    thresholds = _candidate_thresholds(ys, n_grid=n_grid)
    positives = float(yt.sum())
    rows: List[Dict[str, float]] = []
    for threshold in thresholds:
        predicted = ys >= threshold
        tp = float(np.sum(predicted & (yt == 1)))
        fp = float(np.sum(predicted & (yt == 0)))
        fn = positives - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / positives if positives else 0.0
        denominator = precision + recall
        b2 = beta * beta
        rows.append(
            {
                "threshold": float(threshold),
                "precision": precision,
                "recall": recall,
                "f1": (2 * precision * recall / denominator) if denominator else 0.0,
                "fbeta": ((1 + b2) * precision * recall / (b2 * precision + recall))
                if (b2 * precision + recall)
                else 0.0,
                "tp": tp, "fp": fp, "fn": fn,
                "alert_rate": float(predicted.mean()) if predicted.size else 0.0,
            }
        )
    return pd.DataFrame(rows)


def tune_threshold(
    y_true: Sequence[Any],
    y_score: Sequence[Any],
    *,
    objective: str = "fbeta",
    beta: float = DEFAULT_BETA,
    n_grid: int = 201,
    min_precision: Optional[float] = None,
    name: str = "tuned",
) -> Tuple[float, ClassificationReport]:
    """
    Pick the operating point, and return it together with its full report.

    Why 0.5 is the wrong default for this project
    ---------------------------------------------
    0.5 is only the optimal cut when the classes are balanced and the two error
    types cost the same. Neither holds here. The Cresci-2017 graph split is
    1,083 human / 991 bot but the text corpora run far more lopsided, and the
    deployment asymmetry is explicit in the config: ``fusion_model`` records
    that a missed swarm is ~4x worse than a false positive, because a swarm
    that is not flagged keeps operating while a false positive costs one
    analyst one look.

    The naive conclusion is "so maximise recall", and that is wrong too. An
    analyst queue that is mostly noise gets ignored within a week, at which
    point effective recall is zero regardless of what the confusion matrix
    says. F-beta at ``beta=1.5`` (``fusion_model.fbeta``) encodes exactly that
    compromise: recall weighted 2.25x precision in the harmonic mean — enough
    to pull the threshold well below 0.5, not enough to let precision collapse.

    ``objective`` may be any of ``fbeta``, ``f1``, ``precision``, ``recall``,
    ``accuracy``, ``mcc`` or ``youden``. ``min_precision`` adds a hard floor for
    the deployment case where the analyst team has stated a tolerable false
    positive rate and no amount of recall buys a breach of it.
    """
    if objective not in METRIC_NAMES:
        raise ValueError(f"objective must be one of {METRIC_NAMES}, got {objective!r}")

    yt, ys, kind = _prepare(y_true, y_score)
    if kind == "binary":
        log.warning(
            "tune_threshold: y_score is already hard 0/1 — there is nothing to tune. "
            "Returning 0.5 and the report at that point."
        )
        return 0.5, evaluate(yt, ys, threshold=0.5, beta=beta, name=name)

    sweep = threshold_sweep(yt, ys, beta=beta, n_grid=n_grid)
    if min_precision is not None:
        eligible = sweep[sweep["precision"] >= float(min_precision)]
        if eligible.empty:
            log.warning(
                "tune_threshold: no threshold reaches precision >= %.2f (best is %.3f). "
                "Ignoring the floor rather than returning nothing.",
                float(min_precision), float(sweep["precision"].max()),
            )
        else:
            sweep = eligible

    if objective in {"f1", "fbeta", "precision", "recall"}:
        objective_values = sweep[objective].to_numpy(dtype=float)
    elif objective == "youden":
        negatives = float((yt == 0).sum())
        fpr = sweep["fp"].to_numpy(dtype=float) / max(negatives, 1.0)
        objective_values = sweep["recall"].to_numpy(dtype=float) - fpr
    else:
        # accuracy and MCC need the full matrix; they are rarely used here, so
        # paying for a per-candidate evaluate() is acceptable.
        objective_values = np.array(
            [
                getattr(evaluate(yt, ys, threshold=t, beta=beta, name=name), objective)
                for t in sweep["threshold"]
            ],
            dtype=float,
        )

    if not np.isfinite(objective_values).any():
        log.warning("tune_threshold: objective %s is undefined everywhere — falling back to 0.5", objective)
        return 0.5, evaluate(yt, ys, threshold=0.5, beta=beta, name=name)

    best = float(np.nanmax(objective_values))
    # Ties are common on small slices, where many adjacent thresholds sit
    # between the same two scores. Break toward the LOWEST threshold: identical
    # objective, strictly more of the class whose misses cost 4x.
    tied = np.flatnonzero(np.isclose(objective_values, best, rtol=0.0, atol=1e-12))
    chosen = float(sweep["threshold"].to_numpy()[tied[0]])

    report = evaluate(yt, ys, threshold=chosen, beta=beta, name=name)
    log.info(
        "tune_threshold(%s, beta=%.2f): %.4f -> %s=%.4f (0.5 would give %.4f)",
        objective, beta, chosen, objective, best,
        _metric_value(yt, ys, objective, 0.5, beta),
    )
    return chosen, report


# --------------------------------------------------------------------------- #
# Analyst-facing triage
# --------------------------------------------------------------------------- #
def triage_bands(
    scores: Sequence[Any],
    thresholds: Optional[Mapping[str, float]] = None,
) -> pd.Series:
    """
    Map continuous scores onto the ``clear | monitor | investigate | escalate``
    bands from ``fusion_model.decision_thresholds``.

    Bands, not a single cut, because the deliverable is a triage queue rather
    than a verdict. ``escalate`` at 0.82 is small enough to be worked by hand;
    ``monitor`` at 0.35 is deliberately wide, since its purpose is to catch the
    account that only becomes interesting once three of its neighbours also
    light up.

    Returns an ordered ``Categorical`` Series so ``value_counts()``,
    ``sort_values()`` and a groupby all come out in severity order instead of
    alphabetically (which would put ``escalate`` first — actively misleading in
    a report table).
    """
    bands = dict(DEFAULT_BANDS if thresholds is None else thresholds)
    edges = [float(bands[k]) for k in ("monitor", "investigate", "escalate") if k in bands]
    if len(edges) != 3:
        raise ValueError(
            f"thresholds must contain monitor/investigate/escalate, got {sorted(bands)}"
        )
    if not (edges[0] < edges[1] < edges[2]):
        raise ValueError(f"triage thresholds must strictly increase, got {edges}")

    series = scores if isinstance(scores, pd.Series) else pd.Series(np.asarray(scores).ravel())
    numeric = pd.to_numeric(series, errors="coerce")

    unscored = int(numeric.isna().sum())
    if unscored:
        # Left as NaN on purpose: an unscored account must not be silently
        # filed under `clear`, because nobody ever reviews `clear`.
        log.warning(
            "%d row(s) have no usable score and are left unbanded — they will not "
            "appear in any queue. Check the fusion step before shipping the list.",
            unscored,
        )

    banded = pd.cut(
        numeric,
        bins=[-np.inf, *edges, np.inf],
        labels=list(BAND_ORDER),
        right=False,          # score == 0.35 is `monitor`, matching the config's ">=" reading
        ordered=True,
    )
    out = pd.Series(banded, index=series.index, name="triage_band")
    return out.cat.set_categories(list(BAND_ORDER), ordered=True)


# --------------------------------------------------------------------------- #
# Slice analysis
# --------------------------------------------------------------------------- #
def per_group_report(
    y_true: Sequence[Any],
    y_score: Sequence[Any],
    groups: Sequence[Any],
    *,
    threshold: float = 0.5,
    beta: float = DEFAULT_BETA,
    min_support: int = 20,
    sort_by: str = "f1",
) -> pd.DataFrame:
    """
    One full report per group, worst first.

    This produces the most important table in the project. In notebook 02 the
    group is the **generator**, and the question it answers is the only one a
    2026 deployment cares about: the detector saw ``davinci``, ``chatGPT``,
    ``dolly`` and friends in training, but ``cohere`` and ``flant5`` are held
    out (``text_model.holdout_generators``) — does recall survive on a
    generator that was never trained on? An aggregate F1 hides that completely,
    because the held-out generators are a small fraction of the rows. In
    notebook 03 the group is the **relation type**, separating the co-retweet
    ring the model genuinely learned from the co-hashtag edges it may just be
    memorising.

    Groups below ``min_support`` are reported but flagged rather than dropped.
    Dropping them hides an unevaluated slice; reporting them unflagged invites
    someone to quote an F1 computed on 4 rows, where a single flipped
    prediction moves it by 0.25. The flag is a column so it survives
    ``to_markdown()`` into the write-up.
    """
    yt, ys, _ = _prepare(y_true, y_score)
    group_values = pd.Series(np.asarray(groups).ravel()).astype(str).to_numpy()
    if len(group_values) != len(yt):
        raise ValueError(
            f"groups has length {len(group_values)} but {len(yt)} usable rows remain "
            "after cleaning; pass groups aligned to the original y_true"
        )

    rows: List[Dict[str, Any]] = []
    for group in sorted(set(group_values.tolist())):
        positions = np.flatnonzero(group_values == group)
        report = evaluate(
            yt[positions], ys[positions], threshold=threshold, beta=beta, name=str(group)
        )
        record = report.to_dict()
        record["group"] = str(group)
        record["low_support"] = bool(report.support < int(min_support))
        record["single_class"] = bool(report.n_positive in (0, report.support))
        rows.append(record)

    if not rows:
        return pd.DataFrame(columns=["group", "support", "low_support", "f1"])

    frame = pd.DataFrame(rows).drop(columns=["name"])
    ordered = [
        "group", "support", "n_positive", "positive_rate", "low_support", "single_class",
        "precision", "recall", "f1", "fbeta", "accuracy", "roc_auc", "pr_auc", "mcc",
        "brier", "alert_rate", "specificity", "tn", "fp", "fn", "tp", "threshold", "beta",
    ]
    frame = frame.loc[:, [c for c in ordered if c in frame.columns]]

    # Worst first, and low-support groups pushed below well-supported ones with
    # the same score: the eye should land on a real regression, not on noise.
    frame = frame.sort_values(
        by=["low_support", sort_by, "support"], ascending=[True, True, False]
    ).reset_index(drop=True)

    flagged = int(frame["low_support"].sum())
    if flagged:
        log.warning(
            "%d of %d group(s) have fewer than %d rows and are flagged (low_support=True). "
            "Do not quote their F1 without the support column next to it.",
            flagged, len(frame), int(min_support),
        )
    return frame


# --------------------------------------------------------------------------- #
# Uncertainty
# --------------------------------------------------------------------------- #
def _metric_value(
    yt: np.ndarray, ys: np.ndarray, metric: str, threshold: float, beta: float
) -> float:
    """Single scalar, computed without building a full report (bootstrap path)."""
    skm = _sk()
    if metric in {"roc_auc", "pr_auc"}:
        if yt.min() == yt.max():
            return float("nan")
        if metric == "roc_auc":
            return float(skm.roc_auc_score(yt, ys))
        return float(skm.average_precision_score(yt, ys))

    predicted = ys >= threshold
    tp = float(np.sum(predicted & (yt == 1)))
    fp = float(np.sum(predicted & (yt == 0)))
    fn = float(np.sum(~predicted & (yt == 1)))
    tn = float(np.sum(~predicted & (yt == 0)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    if metric == "precision":
        return precision
    if metric == "recall":
        return recall
    if metric == "accuracy":
        total = tp + tn + fp + fn
        return (tp + tn) / total if total else float("nan")
    if metric == "youden":
        return recall - (fp / (fp + tn) if (fp + tn) else 0.0)
    if metric == "mcc":
        return float(skm.matthews_corrcoef(yt, predicted.astype(int)))

    b2 = 1.0 if metric == "f1" else beta * beta
    denominator = b2 * precision + recall
    return (1 + b2) * precision * recall / denominator if denominator else 0.0


def bootstrap_ci(
    y_true: Sequence[Any],
    y_score: Sequence[Any],
    *,
    metric: str = "f1",
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    threshold: float = 0.5,
    beta: float = DEFAULT_BETA,
) -> Tuple[float, float, float]:
    """
    Percentile bootstrap interval, returned as ``(point, lo, hi)``.

    Why the bare point estimate is not reportable
    ---------------------------------------------
    With ``project.smoke_test: true`` every split is capped at 1,500 rows
    (``smoke_test_rows_per_dataset``). On a split that size, with a positive
    class of a few hundred, resampling moves F1 by several points run to run —
    which is the same magnitude as the fusion-over-text-only improvement that
    notebook 04 exists to demonstrate. Quoting "text 0.88 -> fusion 0.91" from
    two point estimates on 1,500 rows is not evidence of anything; quoting
    "0.88 [0.85, 0.91] -> 0.91 [0.88, 0.94]" is at least honest about the fact
    that it might not be.

    The resample is a plain nonparametric bootstrap over rows, seeded, so the
    interval is reproducible. Draws that happen to come out single-class
    contribute NaN for the ranking metrics and are excluded from the
    percentiles (with a warning if they are more than a rounding error) — the
    alternative, a stratified bootstrap, would understate the very
    class-imbalance uncertainty we are trying to measure.
    """
    if metric not in METRIC_NAMES:
        raise ValueError(f"metric must be one of {METRIC_NAMES}, got {metric!r}")

    yt, ys, kind = _prepare(y_true, y_score)
    if yt.size == 0:
        return float("nan"), float("nan"), float("nan")
    if kind == "binary" and metric in {"roc_auc", "pr_auc"}:
        log.warning("bootstrap_ci: %s is undefined for hard 0/1 predictions", metric)
        return float("nan"), float("nan"), float("nan")

    point = _metric_value(yt, ys, metric, threshold, beta)

    rng = np.random.default_rng(seed)
    n = yt.shape[0]
    draws = np.empty(int(n_boot), dtype=float)
    for i in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        draws[i] = _metric_value(yt[idx], ys[idx], metric, threshold, beta)

    degenerate = int(np.isnan(draws).sum())
    if degenerate > 0.01 * n_boot:
        log.warning(
            "bootstrap_ci(%s): %d/%d resamples were degenerate (single-class or empty). "
            "The interval is computed on the rest, but at this rate the split is too "
            "small or too imbalanced for the number to mean much.",
            metric, degenerate, int(n_boot),
        )
    if degenerate == n_boot:
        return point, float("nan"), float("nan")

    lo = float(np.nanpercentile(draws, 100.0 * alpha / 2.0))
    hi = float(np.nanpercentile(draws, 100.0 * (1.0 - alpha / 2.0)))
    return float(point), lo, hi


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def compare_reports(named: Mapping[str, ClassificationReport]) -> pd.DataFrame:
    """
    Ablation table: one row per named report, best F1 first.

    ``delta_f1`` is against the best row rather than against a designated
    baseline, so the table reads the same whether the ablation is
    text-vs-graph-vs-fusion or six variants of the fusion meta-learner.
    """
    if not named:
        return pd.DataFrame()

    frames = []
    for label, report in named.items():
        row = report.to_frame()
        row.index = pd.Index([label], name="model")
        frames.append(row)
    frame = pd.concat(frames)

    columns = [
        "precision", "recall", "f1", "fbeta", "roc_auc", "pr_auc", "mcc", "brier",
        "accuracy", "alert_rate", "threshold", "support", "n_positive",
        "tn", "fp", "fn", "tp",
    ]
    frame = frame.loc[:, [c for c in columns if c in frame.columns]]
    frame = frame.sort_values("f1", ascending=False)
    frame.insert(3, "delta_f1", frame["f1"] - frame["f1"].max())
    return frame


def confusion_frame(
    report: ClassificationReport,
    *,
    labels: Tuple[str, str] = ("human/benign", "adversarial"),
    normalize: bool = False,
) -> pd.DataFrame:
    """
    The 2x2 with row and column totals, labelled in the project's vocabulary.

    Totals are included because the marginals are what make the matrix
    interpretable at a glance: 40 false positives means something very
    different against 200 negatives than against 20,000.
    """
    matrix = np.array(
        [[report.tn, report.fp], [report.fn, report.tp]], dtype=float
    )
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)

    frame = pd.DataFrame(
        matrix,
        index=pd.Index([f"actual {labels[0]}", f"actual {labels[1]}"], name="actual"),
        columns=pd.Index([f"pred {labels[0]}", f"pred {labels[1]}"], name="predicted"),
    )
    if not normalize:
        frame["total"] = frame.sum(axis=1)
        frame.loc["total"] = frame.sum(axis=0)
        frame = frame.astype(int)
    return frame


# --------------------------------------------------------------------------- #
# Config bridges & persistence
# --------------------------------------------------------------------------- #
def bands_from_settings(settings: Settings) -> Dict[str, float]:
    """Read ``fusion_model.decision_thresholds``, falling back to the defaults."""
    configured = settings.section("fusion_model", "decision_thresholds", default=None)
    if not isinstance(configured, dict):
        log.warning("fusion_model.decision_thresholds missing from config — using %s", DEFAULT_BANDS)
        return dict(DEFAULT_BANDS)
    return {str(k): float(v) for k, v in configured.items()}


def beta_from_settings(settings: Settings) -> float:
    """Read ``fusion_model.fbeta``, falling back to :data:`DEFAULT_BETA`."""
    value = settings.section("fusion_model", "fbeta", default=DEFAULT_BETA)
    try:
        return float(value)
    except (TypeError, ValueError):
        log.warning("fusion_model.fbeta is not numeric (%r) — using %.2f", value, DEFAULT_BETA)
        return DEFAULT_BETA


def save_report(
    report: ClassificationReport,
    path: Union[str, Path],
    *,
    extra: Optional[Mapping[str, Any]] = None,
) -> Path:
    """
    Persist a report as JSON next to the model it describes.

    ``extra`` is for the provenance the numbers are meaningless without — the
    dataset name, the split, the commit, whether ``smoke_test`` was on. A
    results JSON that does not say which corpus it came from will be quoted
    against the wrong corpus within a month.
    """
    from .io_utils import save_json

    payload: Dict[str, Any] = {"metrics": report.to_dict()}
    if extra:
        payload["context"] = dict(extra)
    written = save_json(payload, Path(path))
    log.info("wrote %s  %s", written.name, repr(report))
    return written


__all__ = [
    "DEFAULT_BETA", "DEFAULT_BANDS", "BAND_ORDER", "METRIC_NAMES",
    "ClassificationReport",
    "evaluate", "threshold_sweep", "tune_threshold",
    "triage_bands", "per_group_report", "bootstrap_ci",
    "compare_reports", "confusion_frame",
    "bands_from_settings", "beta_from_settings", "save_report",
]
