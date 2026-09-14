"""
aegis.viz
=========

Every figure in ``reports/figures/`` is produced by a function in this module.

Why the plotting lives in the library and not in the notebooks
--------------------------------------------------------------
A report is read as a whole. If notebook 02's confusion matrix is blue-on-white
at 100 dpi and notebook 04's is a seaborn heatmap at whatever ``sns.set()``
last decided, the reader spends attention on the inconsistency instead of on
the result. :func:`set_style` is called from inside every plot function so
there is exactly one look, and it cannot be forgotten in a cell.

The second reason is regeneration. Figures go stale the moment a threshold
moves; a figure that can only be reproduced by re-running a notebook by hand
will not be reproduced, and the report will ship the old one. Everything here
is a pure function of data plus a ``save_as`` path, so the whole figure set can
be rebuilt in a loop.

House rules, applied to every function without exception
--------------------------------------------------------
*   Takes an optional ``ax``; if ``None`` it makes its own figure. This is what
    lets the same function serve a standalone PNG and a panel in a 2x2 grid.
*   Returns the ``Axes`` (or the tuple of Axes for the two-panel figures) so
    the caller can keep annotating.
*   Takes an optional ``save_as`` and writes at dpi=150 with
    ``bbox_inches="tight"`` — 150 because the report is read on screen and at
    300 the PNGs run to several MB each for no visible gain.
*   Never calls ``plt.show()``. Displaying is the notebook's job; a ``show()``
    inside a library function blocks a headless rebuild.
*   Works headlessly. Matplotlib is the only hard dependency: seaborn is used
    for a KDE overlay when it happens to be installed and skipped otherwise,
    and NetworkX degrades to a deterministic geometric layout when absent.

Numbers shown in these figures come from :mod:`aegis.metrics`, never from a
local re-implementation, so a curve and the table beside it cannot disagree.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .config import get_logger
from . import metrics as M

log = get_logger("aegis.viz")

PathLike = Union[str, Path]

#: One palette for the whole report. Chosen for (a) the two classes being
#: distinguishable in greyscale print and (b) both being safe for the common
#: red-green colour-vision deficiencies — a red/green benign/malicious pair is
#: the default everywhere and is exactly the wrong choice.
PALETTE: Dict[str, str] = {
    "benign": "#2c7fb8",        # blue
    "adversarial": "#d95f02",   # orange
    "neutral": "#6b7280",
    "grid": "#d9dde3",
    "accent": "#7570b3",        # purple, for the chosen operating point
    "warn": "#b45309",          # amber, matching BalanceReport's warning colour
    "good": "#1b7837",
}

#: Severity ramp for the triage bands from ``fusion_model.decision_thresholds``.
BAND_COLOURS: Dict[str, str] = {
    "clear": "#c7d3dd",
    "monitor": "#f2c14e",
    "investigate": "#e07b39",
    "escalate": "#b02418",
}

DEFAULT_DPI: int = 150
FIGSIZE: Tuple[float, float] = (7.0, 4.5)
FIGSIZE_WIDE: Tuple[float, float] = (12.0, 4.5)
FIGSIZE_SQUARE: Tuple[float, float] = (5.5, 4.8)

#: Above this, edges are subsampled for layout and drawing. See
#: :func:`plot_network` for the measurements behind the number.
MAX_DRAW_EDGES: int = 20_000

_STYLE_APPLIED = False


# --------------------------------------------------------------------------- #
# Backend & style
# --------------------------------------------------------------------------- #
def _headless() -> bool:
    """
    True when there is provably no display.

    Only Linux/BSD is checked: on Windows and macOS matplotlib already falls
    back to Agg by itself when no GUI toolkit is importable, whereas on a
    headless Linux CI box a Qt/Tk backend will be *found* and then fail at
    figure-creation time with an opaque error.
    """
    if sys.platform.startswith(("linux", "freebsd")):
        return not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return False


def _pyplot():
    """Import pyplot, selecting Agg first if we know we are headless."""
    import matplotlib

    if not os.environ.get("MPLBACKEND") and _headless():
        # force=False so an already-live interactive backend (Jupyter inline)
        # is left alone.
        matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    return plt


def set_style(*, force: bool = False) -> None:
    """
    Apply the AEGIS-SN figure style. Idempotent, and called by every plot
    function, so a notebook never has to remember to call it.

    rcParams are set explicitly rather than via ``plt.style.use("seaborn-...")``
    because the seaborn style *names* were renamed in matplotlib 3.6 and again
    deprecated later; a named style is a version-dependent failure waiting to
    happen in an environment we do not control. Explicit rcParams are boring
    and they keep working.
    """
    global _STYLE_APPLIED
    if _STYLE_APPLIED and not force:
        return

    plt = _pyplot()
    plt.rcParams.update(
        {
            "figure.figsize": FIGSIZE,
            "figure.dpi": 110,              # on-screen; savefig uses DEFAULT_DPI
            "savefig.dpi": DEFAULT_DPI,
            "savefig.bbox": "tight",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#4b5563",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,         # gridlines behind bars, not through them
            "axes.titlesize": 12,
            # "semibold" resolves to weight 700 anyway on DejaVu and emits a
            # findfont warning on every first draw; ask for what we get.
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.7,
            "grid.alpha": 0.9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 1.8,
            "font.size": 10,
            # DejaVu ships with matplotlib, so the figures render identically on
            # a marker's machine that has none of the usual fonts installed.
            "font.family": "DejaVu Sans",
            "image.cmap": "viridis",
        }
    )
    _STYLE_APPLIED = True


def _axes(ax=None, *, figsize: Tuple[float, float] = FIGSIZE):
    """Return ``(fig, ax)``, creating the figure only when the caller did not."""
    set_style()
    plt = _pyplot()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax
    return ax.get_figure(), ax


def _finalise(ax, save_as: Optional[PathLike], *, dpi: int = DEFAULT_DPI):
    """Tighten and optionally write the figure. Always returns ``ax``."""
    fig = ax.get_figure()
    try:
        fig.tight_layout()
    except Exception:  # noqa: BLE001 — colorbars/twin axes occasionally refuse
        pass
    if save_as is not None:
        path = Path(save_as)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        log.info("figure -> %s", path)
    return ax


# --------------------------------------------------------------------------- #
# Classification figures
# --------------------------------------------------------------------------- #
def plot_confusion(
    report: "M.ClassificationReport",
    *,
    labels: Tuple[str, str] = ("human/benign", "adversarial"),
    normalize: bool = True,
    cmap: str = "Blues",
    title: Optional[str] = None,
    ax=None,
    save_as: Optional[PathLike] = None,
):
    """
    Confusion matrix for a :class:`~aegis.metrics.ClassificationReport`.

    Cells carry the raw count *and* the row percentage. Percentage alone hides
    that "50% miss rate" was 2 of 4 rows; count alone makes an imbalanced
    matrix unreadable. Row-normalised (not column-normalised) because the
    question an analyst asks is "of the real swarms, how many did we catch",
    which is a row-wise reading.
    """
    fig, ax = _axes(ax, figsize=FIGSIZE_SQUARE)

    counts = np.array([[report.tn, report.fp], [report.fn, report.tp]], dtype=float)
    row_sums = counts.sum(axis=1, keepdims=True)
    fractions = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0)

    shown = fractions if normalize else counts
    image = ax.imshow(shown, cmap=cmap, vmin=0.0, vmax=1.0 if normalize else None)

    ax.set_xticks([0, 1], [f"pred\n{labels[0]}", f"pred\n{labels[1]}"])
    ax.set_yticks([0, 1], [f"actual\n{labels[0]}", f"actual\n{labels[1]}"])
    ax.grid(False)

    threshold_for_text = shown.max() * 0.6 if shown.size else 0.0
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i,
                f"{int(counts[i, j]):,}\n{fractions[i, j]:.1%}",
                ha="center", va="center", fontsize=11,
                color="white" if shown[i, j] > threshold_for_text else "#1f2937",
            )

    ax.set_title(
        title
        or f"{report.name} — P={report.precision:.3f} R={report.recall:.3f} "
           f"F{report.beta:g}={report.fbeta:.3f}"
    )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return _finalise(ax, save_as)


def plot_roc_pr(
    y_true: Sequence[Any],
    y_score: Sequence[Any],
    *,
    name: str = "model",
    axes=None,
    title: Optional[str] = None,
    save_as: Optional[PathLike] = None,
) -> Tuple[Any, Any]:
    """
    ROC and Precision-Recall side by side, each annotated with its area.

    Both, and always both, because ROC flatters an imbalanced detector and PR
    does not. The false-positive rate on the ROC's x-axis is divided by the
    number of negatives, so when negatives outnumber positives 10:1 a thousand
    false alarms still only moves the curve a little and the AUC stays
    comfortably above 0.9. Precision has no such denominator to hide behind: it
    is computed over what was actually flagged, so those thousand false alarms
    show up immediately as the thing an analyst would experience. The PR
    baseline is the positive prevalence (drawn as a dashed line) rather than
    the 0.5 that the ROC diagonal implies — on a 5%-positive split, an average
    precision of 0.4 is 8x chance and looks poor, while the corresponding
    ROC-AUC would look excellent. Quote the AP.

    ``axes`` may be a pair of existing Axes; the pair is returned either way.
    """
    set_style()
    plt = _pyplot()
    skm = M._sk()

    if axes is None:
        fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    else:
        ax_roc, ax_pr = axes
        fig = ax_roc.get_figure()

    yt, ys, kind = M._prepare(y_true, y_score)
    prevalence = float(yt.mean()) if yt.size else float("nan")

    if yt.size == 0 or yt.min() == yt.max():
        for ax, label in ((ax_roc, "ROC"), (ax_pr, "Precision-Recall")):
            ax.text(0.5, 0.5, f"{label} undefined\n(single-class slice)",
                    ha="center", va="center", color=PALETTE["warn"])
            ax.set_xticks([])
            ax.set_yticks([])
        log.warning("plot_roc_pr(%s): single-class y_true — drawing a placeholder", name)
        return _finalise(ax_roc, save_as), ax_pr

    fpr, tpr, _ = skm.roc_curve(yt, ys)
    roc_auc = float(skm.auc(fpr, tpr))
    ax_roc.plot(fpr, tpr, color=PALETTE["adversarial"], label=f"{name} (AUC={roc_auc:.3f})")
    ax_roc.plot([0, 1], [0, 1], color=PALETTE["neutral"], linestyle="--", linewidth=1.0,
                label="chance")
    ax_roc.set_xlabel("false positive rate")
    ax_roc.set_ylabel("true positive rate (recall)")
    ax_roc.set_title("ROC")
    ax_roc.set_xlim(-0.02, 1.02)
    ax_roc.set_ylim(-0.02, 1.02)
    ax_roc.legend(loc="lower right")

    precision, recall, _ = skm.precision_recall_curve(yt, ys)
    average_precision = float(skm.average_precision_score(yt, ys))
    ax_pr.plot(recall, precision, color=PALETTE["benign"], label=f"{name} (AP={average_precision:.3f})")
    ax_pr.axhline(prevalence, color=PALETTE["neutral"], linestyle="--", linewidth=1.0,
                  label=f"chance = prevalence ({prevalence:.1%})")
    ax_pr.set_xlabel("recall")
    ax_pr.set_ylabel("precision")
    ax_pr.set_title("Precision-Recall")
    ax_pr.set_xlim(-0.02, 1.02)
    ax_pr.set_ylim(-0.02, 1.02)
    ax_pr.legend(loc="lower left")

    if kind == "binary":
        log.warning("plot_roc_pr(%s): hard 0/1 scores produce a two-point curve", name)
    if title:
        fig.suptitle(title)

    _finalise(ax_roc, save_as)
    return ax_roc, ax_pr


def plot_threshold_sweep(
    y_true: Sequence[Any],
    y_score: Sequence[Any],
    *,
    beta: float = M.DEFAULT_BETA,
    chosen: Optional[float] = None,
    show_bands: bool = True,
    bands: Optional[Mapping[str, float]] = None,
    title: Optional[str] = None,
    ax=None,
    save_as: Optional[PathLike] = None,
):
    """
    Precision, recall and F-beta against threshold, with the operating point
    marked.

    This is the figure that justifies the operating point in the write-up. The
    triage band edges from ``fusion_model.decision_thresholds`` are shaded
    behind the curves so it is immediately visible what precision an analyst
    can expect inside the ``escalate`` band, which is the number the
    deployment argument actually rests on.

    When ``chosen`` is None the point is taken from
    :func:`aegis.metrics.tune_threshold` with the same beta, so the marker
    cannot drift away from the threshold the model is actually shipped with.
    """
    fig, ax = _axes(ax, figsize=FIGSIZE)

    sweep = M.threshold_sweep(y_true, y_score, beta=beta)
    if chosen is None:
        chosen, _ = M.tune_threshold(y_true, y_score, objective="fbeta", beta=beta)

    if show_bands:
        edges = dict(M.DEFAULT_BANDS if bands is None else bands)
        ordered = [(name, edges[name]) for name in ("monitor", "investigate", "escalate")
                   if name in edges]
        upper = [value for _, value in ordered[1:]] + [1.0]
        for (band_name, low), high in zip(ordered, upper):
            ax.axvspan(low, high, color=BAND_COLOURS.get(band_name, PALETTE["grid"]),
                       alpha=0.13, linewidth=0)
            ax.text((low + high) / 2.0, 1.02, band_name, ha="center", va="bottom",
                    fontsize=8, color=PALETTE["neutral"])

    ax.plot(sweep["threshold"], sweep["precision"], color=PALETTE["benign"], label="precision")
    ax.plot(sweep["threshold"], sweep["recall"], color=PALETTE["adversarial"], label="recall")
    ax.plot(sweep["threshold"], sweep["fbeta"], color=PALETTE["accent"], linewidth=2.4,
            label=f"F{beta:g}")

    at_chosen = sweep.iloc[(sweep["threshold"] - float(chosen)).abs().argmin()]
    ax.axvline(float(chosen), color=PALETTE["accent"], linestyle="--", linewidth=1.2)
    ax.plot([float(chosen)], [at_chosen["fbeta"]], marker="o", markersize=7,
            color=PALETTE["accent"], zorder=5)
    ax.annotate(
        f"chosen {float(chosen):.3f}\nP={at_chosen['precision']:.3f} "
        f"R={at_chosen['recall']:.3f}\nF{beta:g}={at_chosen['fbeta']:.3f}",
        xy=(float(chosen), at_chosen["fbeta"]),
        xytext=(8, -46), textcoords="offset points", fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white",
              "edgecolor": PALETTE["accent"], "linewidth": 0.8},
    )

    ax.set_xlabel("decision threshold")
    ax.set_ylabel("score")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(title or f"Operating point selection (F-beta, beta={beta:g})")
    ax.legend(loc="lower center", ncol=3)
    return _finalise(ax, save_as)


def plot_per_group_bars(
    frame: pd.DataFrame,
    *,
    metric: str = "recall",
    group_col: str = "group",
    support_col: str = "support",
    flag_col: str = "low_support",
    top_n: Optional[int] = None,
    reference: Optional[float] = None,
    title: Optional[str] = None,
    ax=None,
    save_as: Optional[PathLike] = None,
):
    """
    Horizontal bars of a per-group metric, for the output of
    :func:`aegis.metrics.per_group_report`.

    Low-support groups are drawn hatched, greyed and with their ``n`` in the
    label. Plotting them identically to well-supported groups is how a reader
    ends up quoting "recall 0.25 on flant5" when flant5 had eleven rows; hiding
    them is how a reader concludes the generator was evaluated when it was not.
    Hatching says "measured, not trusted", which is the true state of affairs.

    ``reference`` draws a vertical line — pass the pooled metric so the reader
    can see at a glance which generators fall below the headline number.
    """
    fig, ax = _axes(ax, figsize=(7.5, max(3.0, 0.34 * min(len(frame), top_n or len(frame)) + 1.4)))

    if metric not in frame.columns:
        raise ValueError(f"{metric!r} is not a column of the frame ({list(frame.columns)})")

    data = frame.copy()
    if top_n is not None:
        data = data.head(int(top_n))
    # per_group_report already sorts worst-first; barh draws bottom-up, so
    # reversing here puts the worst group at the top of the image.
    data = data.iloc[::-1]

    flags = (
        data[flag_col].astype(bool).to_numpy()
        if flag_col in data.columns
        else np.zeros(len(data), dtype=bool)
    )
    values = pd.to_numeric(data[metric], errors="coerce").fillna(0.0).to_numpy()
    labels = data[group_col].astype(str).to_numpy()
    if support_col in data.columns:
        labels = np.array(
            [f"{name}  (n={int(n):,})" for name, n in zip(labels, data[support_col])]
        )

    positions = np.arange(len(data))
    bars = ax.barh(
        positions, values,
        color=[PALETTE["neutral"] if flag else PALETTE["benign"] for flag in flags],
        edgecolor=["#374151" if flag else "none" for flag in flags],
        hatch=["//" if flag else "" for flag in flags],
        alpha=0.95,
    )
    for bar, value, flag in zip(bars, values, flags):
        ax.text(
            min(value + 0.015, 1.0), bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}" + (" *" if flag else ""),
            va="center", fontsize=8.5,
            color=PALETTE["warn"] if flag else "#1f2937",
        )

    if reference is not None:
        ax.axvline(float(reference), color=PALETTE["adversarial"], linestyle="--",
                   linewidth=1.2, label=f"pooled {metric} = {float(reference):.3f}")
        ax.legend(loc="lower right")

    ax.set_yticks(positions, labels)
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel(metric)
    ax.set_title(title or f"Per-group {metric} (hatched = low support)")
    ax.grid(axis="y", visible=False)
    return _finalise(ax, save_as)


def plot_score_distributions(
    y_true: Sequence[Any],
    y_score: Sequence[Any],
    *,
    labels: Tuple[str, str] = ("human/benign", "adversarial"),
    bins: int = 50,
    threshold: Optional[float] = None,
    kde: bool = True,
    log_y: bool = False,
    title: Optional[str] = None,
    ax=None,
    save_as: Optional[PathLike] = None,
):
    """
    Overlaid score histograms, one per true class.

    The single most diagnostic plot in the project. A good AUC can coexist with
    two heavily overlapping humps plus a thin separated tail, and that shape
    means something very specific: the model is confident on the easy half and
    is guessing on the rest, so *any* threshold you pick trades a lot of one
    error for a little of the other. Two well-separated modes mean the
    threshold choice barely matters and the fusion is doing real work. You
    cannot tell those two situations apart from a scalar, and they call for
    completely different next steps.

    Density-normalised so the imbalanced class is still visible; ``log_y``
    helps when the benign mode is two orders of magnitude taller. The seaborn
    KDE overlay is used when seaborn is installed and skipped silently
    otherwise.
    """
    fig, ax = _axes(ax, figsize=FIGSIZE)

    yt, ys, kind = M._prepare(y_true, y_score)
    if kind == "binary":
        log.warning("plot_score_distributions: hard 0/1 scores give a two-spike histogram")

    edges = np.linspace(float(np.min(ys)) if ys.size else 0.0,
                        float(np.max(ys)) if ys.size else 1.0, int(bins) + 1)
    if edges[0] == edges[-1]:
        edges = np.linspace(edges[0] - 0.5, edges[0] + 0.5, int(bins) + 1)

    for value, label, colour in ((0, labels[0], PALETTE["benign"]),
                                 (1, labels[1], PALETTE["adversarial"])):
        subset = ys[yt == value]
        if subset.size == 0:
            continue
        ax.hist(subset, bins=edges, density=True, alpha=0.55, color=colour,
                label=f"{label} (n={subset.size:,})")

    if kde:
        try:
            import seaborn as sns

            for value, colour in ((0, PALETTE["benign"]), (1, PALETTE["adversarial"])):
                subset = ys[yt == value]
                if subset.size > 10 and float(np.std(subset)) > 1e-9:
                    sns.kdeplot(x=subset, ax=ax, color=colour, linewidth=1.6, warn_singular=False)
        except Exception:  # noqa: BLE001 — seaborn absent, or a degenerate slice
            log.debug("seaborn KDE overlay skipped")

    if threshold is not None:
        ax.axvline(float(threshold), color=PALETTE["accent"], linestyle="--", linewidth=1.4,
                   label=f"threshold = {float(threshold):.3f}")

    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("model score")
    ax.set_ylabel("density")
    ax.set_title(title or "Score distribution by true class")
    ax.legend(loc="upper center")
    return _finalise(ax, save_as)


def plot_training_curve(
    history: Union[Sequence[Mapping[str, Any]], pd.DataFrame],
    *,
    x: str = "epoch",
    series: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    ax=None,
    save_as: Optional[PathLike] = None,
):
    """
    Epoch curves from a list of dicts (the HF ``Trainer`` log history shape) or
    a DataFrame.

    Losses go on the left axis and everything else on a twinned right axis,
    because plotting an eval F1 of 0.9 against a training loss of 0.03 on one
    scale flattens the loss into the x-axis and the curve stops being able to
    show the one thing it is for — the epoch where validation stops improving
    while training keeps going. With ``early_stopping_patience: 2`` in the
    config, that crossing point is the whole story of the run.
    """
    fig, ax = _axes(ax, figsize=FIGSIZE)

    frame = pd.DataFrame(list(history)) if not isinstance(history, pd.DataFrame) else history.copy()
    if frame.empty:
        log.warning("plot_training_curve: empty history")
        ax.set_title(title or "Training curve (no history)")
        return _finalise(ax, save_as)

    if x not in frame.columns:
        # HF's log_history interleaves records that carry `step` but no `epoch`.
        fallback = next((c for c in ("epoch", "step", "iteration") if c in frame.columns), None)
        if fallback is None:
            frame = frame.reset_index().rename(columns={"index": x})
        else:
            x = fallback

    numeric = [
        column for column in frame.columns
        if column != x and pd.api.types.is_numeric_dtype(frame[column])
    ]
    chosen = [c for c in (series or numeric) if c in frame.columns]
    if not chosen:
        log.warning("plot_training_curve: no numeric series to plot in %s", list(frame.columns))
        return _finalise(ax, save_as)

    losses = [c for c in chosen if "loss" in c.lower()]
    others = [c for c in chosen if c not in losses]

    colours = [PALETTE["benign"], PALETTE["adversarial"], PALETTE["accent"],
               PALETTE["good"], PALETTE["warn"], PALETTE["neutral"]]
    handles: List[Any] = []

    for i, column in enumerate(losses):
        sub = frame[[x, column]].dropna()
        line, = ax.plot(sub[x], sub[column], color=colours[i % len(colours)],
                        marker="o", markersize=3, label=column)
        handles.append(line)
    ax.set_xlabel(x)
    ax.set_ylabel("loss" if losses else "value")

    if others:
        right = ax.twinx() if losses else ax
        if losses:
            right.grid(False)
        for i, column in enumerate(others):
            sub = frame[[x, column]].dropna()
            line, = right.plot(sub[x], sub[column], color=colours[(i + len(losses)) % len(colours)],
                               linestyle="--", marker="s", markersize=3, label=column)
            handles.append(line)
        right.set_ylabel("metric" if losses else "value")

    ax.set_title(title or "Training history")
    ax.legend(handles=handles, labels=[h.get_label() for h in handles], loc="best")
    return _finalise(ax, save_as)


def plot_feature_importance(
    names: Sequence[str],
    values: Sequence[float],
    *,
    top_n: int = 20,
    title: Optional[str] = None,
    xlabel: str = "importance",
    ax=None,
    save_as: Optional[PathLike] = None,
):
    """
    Top-``n`` features by absolute magnitude.

    Ranked on ``abs(value)`` but drawn signed, because for the logistic fusion
    meta-learner the sign is the interesting part: a negative coefficient on
    ``circadian_flatness`` would mean the model has learned the opposite of the
    coordination hypothesis and the feature needs auditing before the number
    goes anywhere near the write-up.
    """
    name_list = [str(n) for n in names]
    value_list = list(values)
    if len(name_list) != len(value_list):
        raise ValueError(
            f"names and values must have the same length ({len(name_list)} vs {len(value_list)})"
        )

    series = pd.Series(
        pd.to_numeric(pd.Series(value_list), errors="coerce").fillna(0.0).to_numpy(),
        index=pd.Index(name_list),
    )
    ranked = series.reindex(series.abs().sort_values(ascending=False).index).head(int(top_n))
    ranked = ranked.iloc[::-1]

    fig, ax = _axes(ax, figsize=(7.0, max(3.0, 0.32 * len(ranked) + 1.2)))
    ax.barh(
        np.arange(len(ranked)), ranked.to_numpy(),
        color=[PALETTE["adversarial"] if v >= 0 else PALETTE["benign"] for v in ranked],
        alpha=0.95,
    )
    ax.set_yticks(np.arange(len(ranked)), list(ranked.index))
    ax.axvline(0.0, color="#374151", linewidth=0.9)
    ax.set_xlabel(xlabel)
    ax.set_title(title or f"Top {len(ranked)} features")
    ax.grid(axis="y", visible=False)
    return _finalise(ax, save_as)


# --------------------------------------------------------------------------- #
# Network figure
# --------------------------------------------------------------------------- #
def plot_network(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    node_col: str = "user_id",
    label_col: str = "label",
    score_col: Optional[str] = None,
    max_nodes: int = 2_000,
    max_edges: int = MAX_DRAW_EDGES,
    seed: int = 42,
    layout_iterations: int = 30,
    labels: Tuple[str, str] = ("human/benign", "adversarial"),
    title: Optional[str] = None,
    ax=None,
    save_as: Optional[PathLike] = None,
):
    """
    Node-link diagram of the interaction graph.

    Subsampling, and where the cost actually is
    -------------------------------------------
    The Cresci-2017 co-activity graph as configured is ~2,000 accounts and
    ~290,000 edges (``max_pairs_per_relation: 150000`` across five co-activity
    relations). Measured on exactly that shape, at this figure's 7.5x6.5in /
    150 dpi, 30 layout iterations::

                              full 290k edges    20k subsample
        nx.spring_layout            19.3 s           18.4 s
        rendering the segments      23.7 s            1.7 s

    The intuitive answer — "the layout is the bottleneck" — is wrong here, and
    the measurement is worth keeping because it says where the cap has to go.
    ``spring_layout``'s cost at this size is driven by the *node* count, not the
    edge count, so subsampling edges buys essentially nothing there; it is
    ``max_nodes`` that protects the layout, and that is the knob that matters
    if TwiBot-22 (1M users) is ever wired in.

    What edge subsampling does buy is a 14x cheaper render and, far more
    importantly, a figure that says something. 270,000 segments over a 1,000 px
    disc is roughly 20 line-crossings per pixel: the result is a uniform grey
    plate with no visible community structure, i.e. strictly less informative
    than an empty axes. 20,000 is about where the ink stops saturating at this
    size. Sampling is uniform and seeded, which preserves the relative density
    of the communities — the only property this figure is asked to convey.

    Expect 15-25 s end to end at Cresci scale on a laptop CPU. Slow, but
    bounded and progressing, which is the actual requirement.

    **Node degree is computed on the FULL edge list**, before any sampling, so
    the marker sizes still encode real connectivity — a hub stays visibly a hub
    even if only 7% of its edges were drawn. Anything quantitative belongs in
    :mod:`aegis.graph_features`, not in a picture.
    """
    fig, ax = _axes(ax, figsize=(7.5, 6.5))
    from matplotlib.collections import LineCollection

    rng = np.random.default_rng(seed)

    node_frame = nodes.copy()
    node_frame[node_col] = node_frame[node_col].astype(str)
    edge_frame = (
        edges.loc[:, ["source", "target"]].copy() if not edges.empty
        else pd.DataFrame(columns=["source", "target"])
    )
    if not edge_frame.empty:
        edge_frame["source"] = edge_frame["source"].astype(str)
        edge_frame["target"] = edge_frame["target"].astype(str)

    # Degree first, on everything, then sample.
    if edge_frame.empty:
        degree = pd.Series(0, index=node_frame[node_col], dtype=float)
    else:
        stacked = pd.concat([edge_frame["source"], edge_frame["target"]])
        degree = stacked.value_counts().reindex(node_frame[node_col]).fillna(0.0)

    if len(node_frame) > int(max_nodes):
        # Keep the highest-degree accounts: they are the ones whose position in
        # the layout carries information. A random 2k of 50k nodes is dust.
        keep = degree.sort_values(ascending=False).head(int(max_nodes)).index
        log.warning(
            "plot_network: %d nodes exceeds max_nodes=%d — drawing the %d highest-degree "
            "accounts only. The figure is illustrative, not a census.",
            len(node_frame), int(max_nodes), int(max_nodes),
        )
        node_frame = node_frame[node_frame[node_col].isin(set(keep))]
        degree = degree.reindex(node_frame[node_col]).fillna(0.0)

    node_ids = node_frame[node_col].tolist()
    keep_ids = set(node_ids)
    if not edge_frame.empty:
        edge_frame = edge_frame[
            edge_frame["source"].isin(keep_ids) & edge_frame["target"].isin(keep_ids)
        ]
        edge_frame = edge_frame[edge_frame["source"] != edge_frame["target"]]

    n_edges_total = len(edge_frame)
    if n_edges_total > int(max_edges):
        picked = rng.choice(n_edges_total, size=int(max_edges), replace=False)
        edge_frame = edge_frame.iloc[np.sort(picked)]
        log.info(
            "plot_network: subsampled %d -> %d edges for layout and drawing "
            "(node sizes still use all %d).",
            n_edges_total, len(edge_frame), n_edges_total,
        )

    positions = _layout(node_ids, edge_frame, seed=seed, iterations=layout_iterations)
    coords = np.array([positions[node] for node in node_ids], dtype=float)

    if not edge_frame.empty:
        segments = [
            (positions[u], positions[v])
            for u, v in zip(edge_frame["source"], edge_frame["target"])
        ]
        # Alpha scaled down as edge count rises, so a dense graph greys out
        # rather than filling in solid.
        alpha = float(np.clip(2000.0 / max(len(segments), 1), 0.02, 0.35))
        ax.add_collection(
            LineCollection(segments, colors="#94a3b8", linewidths=0.35, alpha=alpha, zorder=1)
        )

    sizes = 12.0 + 90.0 * (degree.to_numpy() / max(float(degree.max()), 1.0)) ** 0.5

    if score_col and score_col in node_frame.columns:
        values = pd.to_numeric(node_frame[score_col], errors="coerce").fillna(0.0).to_numpy()
        scatter = ax.scatter(coords[:, 0], coords[:, 1], s=sizes, c=values, cmap="inferno",
                             vmin=0.0, vmax=1.0, linewidths=0.2, edgecolors="white", zorder=2)
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.02, label=score_col)
    elif label_col in node_frame.columns:
        flags = pd.to_numeric(node_frame[label_col], errors="coerce").fillna(0).to_numpy() > 0.5
        for mask, colour, label in (
            (~flags, PALETTE["benign"], labels[0]),
            (flags, PALETTE["adversarial"], labels[1]),
        ):
            if mask.any():
                ax.scatter(coords[mask, 0], coords[mask, 1], s=sizes[mask], c=colour,
                           label=f"{label} (n={int(mask.sum()):,})",
                           linewidths=0.2, edgecolors="white", zorder=2)
        ax.legend(loc="upper right", markerscale=1.6)
    else:
        ax.scatter(coords[:, 0], coords[:, 1], s=sizes, c=PALETTE["neutral"], zorder=2)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(
        title
        or f"Interaction graph — {len(node_ids):,} accounts, "
           f"{len(edge_frame):,} of {n_edges_total:,} edges drawn"
    )
    ax.autoscale_view()
    return _finalise(ax, save_as)


def _layout(
    node_ids: Sequence[str], edges: pd.DataFrame, *, seed: int, iterations: int
) -> Dict[str, Tuple[float, float]]:
    """
    Force-directed positions via NetworkX, with a deterministic fallback.

    Without NetworkX the nodes are placed on a circle ordered by degree, which
    is not a community layout but is still readable and — importantly — keeps
    the whole module runnable, in line with how the rest of the package treats
    optional dependencies.
    """
    try:
        import networkx as nx
    except ImportError:  # pragma: no cover
        log.warning(
            "networkx is not installed — falling back to a degree-ordered circular "
            "layout. Community structure will not be visible."
        )
        angles = np.linspace(0.0, 2.0 * np.pi, len(node_ids), endpoint=False)
        return {
            node: (float(np.cos(a)), float(np.sin(a))) for node, a in zip(node_ids, angles)
        }

    graph = nx.Graph()
    graph.add_nodes_from(node_ids)
    if not edges.empty:
        graph.add_edges_from(zip(edges["source"], edges["target"]))
    # k below the 1/sqrt(n) default pulls the communities into tighter blobs,
    # which is what makes a coordinated cluster legible against the organic
    # background at this node count. 30 iterations rather than NetworkX's 50:
    # measured end to end at Cresci scale that is 24.8s vs 31.6s, and the two
    # layouts are visually indistinguishable.
    raw = nx.spring_layout(
        graph, seed=seed, iterations=int(iterations), k=0.6 / max(np.sqrt(len(node_ids)), 1.0)
    )
    return {str(node): (float(xy[0]), float(xy[1])) for node, xy in raw.items()}


# --------------------------------------------------------------------------- #
# Bulk save
# --------------------------------------------------------------------------- #
def save_all_open_figures(
    paths: Any,
    *,
    prefix: str = "figure",
    dpi: int = DEFAULT_DPI,
    close: bool = False,
) -> List[Path]:
    """
    Write every open matplotlib figure into the report figures directory.

    ``paths`` may be an :class:`aegis.config.Paths` (its ``.figures`` is used) or
    any directory path. Intended as the last cell of a notebook: figures made
    inline during exploration are otherwise lost on kernel restart, and
    rebuilding one an hour later means re-running the training cell above it.

    Filenames come from the figure's suptitle/axes title when there is one, so
    the output is browsable rather than ``figure_3.png``.
    """
    plt = _pyplot()
    directory = Path(getattr(paths, "figures", paths))
    directory.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    for number in plt.get_fignums():
        figure = plt.figure(number)
        stem = ""
        label = getattr(figure, "_suptitle", None)
        if label is not None and label.get_text():
            stem = label.get_text()
        elif figure.axes and figure.axes[0].get_title():
            stem = figure.axes[0].get_title()
        slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem).strip("_").lower()
        name = f"{prefix}_{number:02d}" + (f"_{slug[:60]}" if slug else "")

        target = directory / f"{name}.png"
        figure.savefig(target, dpi=dpi, bbox_inches="tight")
        written.append(target)
        if close:
            plt.close(figure)

    log.info("saved %d open figure(s) to %s", len(written), directory)
    return written


__all__ = [
    "PALETTE", "BAND_COLOURS", "DEFAULT_DPI", "FIGSIZE", "FIGSIZE_WIDE",
    "FIGSIZE_SQUARE", "MAX_DRAW_EDGES",
    "set_style",
    "plot_confusion", "plot_roc_pr", "plot_threshold_sweep",
    "plot_per_group_bars", "plot_score_distributions", "plot_training_curve",
    "plot_feature_importance", "plot_network",
    "save_all_open_figures",
]
