"""
Generate the publication-quality plots requested in the second review round:

  1. Precision-Recall curve (the central reference under class imbalance)
  2. Precision, Recall, F1 as a function of the decision threshold
  3. Cost as a function of the decision threshold, for several C_FN:C_FP ratios
  4. (Optional) PR curves stratified by LLM-judge confidence buckets

All inputs are CSVs produced by earlier pipeline steps; no model is retrained
here. Figures are written under artifacts/figures/.

Run from repo root:
    python scripts/make_plots.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display required
import matplotlib.pyplot as plt

# Paper-grade defaults: larger fonts, no in-figure title (caption goes in LaTeX),
# tight spacing, vector + high-DPI raster output.
DPI = 300
plt.rcParams.update({
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "legend.fontsize":   9,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "figure.dpi":       100,    # screen rendering; savefig overrides for output
    "savefig.dpi":      DPI,
    "savefig.bbox":     "tight",
    "pdf.fonttype":     42,     # editable text in PDF (TrueType, not bitmaps)
    "ps.fonttype":      42,
})


def _save(fig, path: Path) -> None:
    """Write each figure as both PNG (300 dpi) and PDF (vector)."""
    fig.savefig(path)                               # PNG
    fig.savefig(path.with_suffix(".pdf"))           # PDF (vector)
    plt.close(fig)
    print(f"  → {path}  +  {path.with_suffix('.pdf').name}")

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from td_prediction import config
from td_prediction.data.splits import FeatureSet, Split, load_split, prepare_xy
from td_prediction.models import trainer

# inputs
PR_CURVE   = Path("artifacts/results/pr_curve.csv")
OPS_CSV    = Path("artifacts/results/operating_points.csv")
COST_CSV   = Path("artifacts/results/cost_curve.csv")
PARSED_LBL = Path("artifacts/results/parsed_labels_v2_rubric_json.csv")
FEATURES   = Path("data/features_with_llm_labels.csv")

# outputs
FIG_DIR = Path("artifacts/figures")
FIG_PR             = FIG_DIR / "pr_curve.png"
FIG_THRESHOLD      = FIG_DIR / "threshold_sweep.png"
FIG_COST           = FIG_DIR / "cost_vs_threshold.png"
FIG_PR_STRATIFIED        = FIG_DIR / "pr_curve_by_confidence.png"
FIG_THRESHOLD_STRATIFIED = FIG_DIR / "threshold_sweep_by_confidence.png"
FIG_DECISION_BEHAVIOR    = FIG_DIR / "decision_behavior.png"

LABEL_COL = "label_consolidated"


# ── 1. PR curve ────────────────────────────────────────────────────────────────

def plot_pr_curve(curve: pd.DataFrame, ops: pd.DataFrame) -> None:
    """Precision vs recall, with three operating points marked.

    Kept lean for paper use: F1-optimal (the headline), cost-optimal at 10:1
    (the realistic deployment point), and the default 0.5 threshold for
    comparison. F0.5/F2/cost-1:1/cost-5:1 omitted to reduce clutter; they
    remain in operating_points.csv for the curious reader.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    # Single curve → no need to differentiate by colour; black is cleanest.
    ax.plot(curve["recall"], curve["precision"], lw=2, color="black",
            label="Random Forest (all features, no rebalancing)")

    # baseline = test-set positive rate
    pos_rate = 0.0537
    ax.axhline(pos_rate, ls="--", color="grey", alpha=0.7,
               label=f"random baseline (positive rate = {pos_rate:.3f})")

    # Operating points differ by SHAPE (not colour); shape carries the meaning.
    style = {
        "f1_optimal":        ("o", "F1-optimal"),
        "cost_optimal_10:1": ("D", "Cost-optimal (10:1)"),
        "default_0.5":       ("s", "Default threshold 0.5"),
    }
    for _, op in ops.iterrows():
        if op["name"] in style:
            marker, label = style[op["name"]]
            ax.scatter(op["recall"], op["precision"], marker=marker, s=110,
                       color="white", label=label, edgecolors="black",
                       linewidths=1.4, zorder=5)

    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, FIG_PR)


# ── 2. P / R / F1 vs threshold ─────────────────────────────────────────────────

def plot_threshold_sweep(curve: pd.DataFrame, ops: pd.DataFrame) -> None:
    """How P, R, F1 evolve as the decision threshold changes.

    Three lines distinguished by both colour and line style so the plot remains
    legible in greyscale or under colour-vision deficiency.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(curve["threshold"], curve["precision"], lw=2, ls="-",
            color="#1f77b4", label="Precision")
    ax.plot(curve["threshold"], curve["recall"],    lw=2, ls="--",
            color="#ff7f0e", label="Recall")
    ax.plot(curve["threshold"], curve["f1"],        lw=2, ls="-.",
            color="#2ca02c", label="F1")

    # Two vertical guides only — keep the chart readable
    for name, ls, label in [
        ("f1_optimal",       ":",   "F1-optimal"),
        ("cost_optimal_10:1","--",  "Cost-optimal (10:1)"),
    ]:
        match = ops[ops["name"] == name]
        if len(match):
            t = float(match.iloc[0]["threshold"])
            ax.axvline(t, ls=ls, color="black", alpha=0.6, lw=1.2,
                       label=f"{label} (τ={t:.2f})")

    ax.set_xlabel("Decision threshold τ"); ax.set_ylabel("Metric value")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="center right", framealpha=0.95)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, FIG_THRESHOLD)


# ── 3. Cost vs threshold ───────────────────────────────────────────────────────

def plot_cost_curve(cost_df: pd.DataFrame, ops: pd.DataFrame) -> None:
    """Total absolute cost vs decision threshold, one subplot per ratio.

    Absolute (not normalised) so reviewers can read off the actual cost units.
    Each panel has its own y-scale because the three ratios differ by ~10×
    in magnitude and would visually swamp each other on a shared axis.
    """
    ratios = ["1:1", "5:1", "10:1"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True)
    for ax, ratio in zip(axes, ratios):
        sub = cost_df[cost_df["ratio"] == ratio].sort_values("threshold")
        # All three panels use the same colour — they are separated by panel,
        # not by colour, so colour would be decorative.
        ax.plot(sub["threshold"], sub["cost"], lw=2, color="black")

        op_name = f"cost_optimal_{ratio}"
        match = ops[ops["name"] == op_name]
        if len(match):
            t = float(match.iloc[0]["threshold"])
            min_cost = float(sub["cost"].min())
            ax.axvline(t, ls=":", color="black", alpha=0.5, lw=1.0)
            ax.scatter([t], [min_cost], s=110, color="white",
                       edgecolors="black", linewidths=1.4, zorder=5,
                       label=f"optimum: τ={t:.2f}, cost={min_cost:.0f}")
            ax.legend(loc="upper right", framealpha=0.95)

        # In-axes label of the ratio (replaces a chart title)
        ax.text(0.04, 0.94, f"$C_{{FN}}$ : $C_{{FP}} = {ratio}$",
                transform=ax.transAxes, fontsize=11, fontweight="bold",
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

        ax.set_xlabel("Decision threshold τ")
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Total misclassification cost")
    fig.tight_layout()
    _save(fig, FIG_COST)


# ── 4. PR curves stratified by LLM-judge confidence ────────────────────────────

def _refresh_labels(split, labels_df):
    cols = [c for c in ("commit_uid", "label_llm", "label_satd", "label_consolidated", "label_human")
            if c in labels_df.columns]
    fresh = labels_df[cols]
    def _merge(df):
        df = df.drop(columns=[c for c in cols if c != "commit_uid" and c in df.columns])
        return df.merge(fresh, on="commit_uid", how="left")
    return Split(train=_merge(split.train), val=_merge(split.val),
                 test=_merge(split.test), name=split.name)


def _train_and_score_for_stratification():
    """Shared model training + scoring used by stratified and combined plots."""
    print("  Re-training RF / all / none for stratified curves...")
    labels_df = pd.read_csv(FEATURES)
    parsed    = pd.read_csv(PARSED_LBL)[["commit_uid", "confidence"]]
    split = _refresh_labels(load_split(config.PATHS.splits, "time"), labels_df)

    bundle = trainer.train_bundle(
        split=split, model_kind="rf", imbalance="none",
        feature_set=FeatureSet.ALL, label_col=LABEL_COL,
        threshold_objective="f1",
    )

    test_with_conf = split.test.merge(parsed, on="commit_uid", how="left")
    test_with_conf["confidence"] = test_with_conf["confidence"].fillna(
        test_with_conf["confidence"].median()
    )
    X_te, y_te, _, _ = prepare_xy(
        test_with_conf, label_col=LABEL_COL,
        feature_set=FeatureSet.ALL, feature_cols=bundle.feature_cols,
    )
    return {
        "y_score":   bundle.model.predict_proba(X_te)[:, 1],
        "y_true":    y_te.to_numpy(),
        "conf":      test_with_conf["confidence"].to_numpy(),
    }


def plot_stratified_by_confidence(scoring=None) -> None:
    """Two stratified plots: PR curve and P/R/F1-vs-threshold per confidence bucket."""
    if scoring is None:
        scoring = _train_and_score_for_stratification()
    y_score = scoring["y_score"]
    y_true  = scoring["y_true"]
    conf    = scoring["conf"]

    # Each bucket: (label, mask, colour, line style). Colour and line style
    # together so the plot survives greyscale printing.
    buckets = [
        ("low (<0.85)",      conf < 0.85,                      "#d62728", ":"),
        ("mid (0.85–0.95)", (conf >= 0.85) & (conf < 0.95),    "#ff7f0e", "--"),
        ("high (≥0.95)",     conf >= 0.95,                     "#2ca02c", "-"),
    ]

    # 1) Stratified PR curve (same axes for direct comparison) ---------------
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, mask, color, ls in buckets:
        n = int(mask.sum()); n_pos = int(y_true[mask].sum())
        if n_pos == 0 or n_pos == n:
            print(f"  PR: skipping bucket '{label}': n={n}, positives={n_pos} (no positives)")
            continue
        p_arr, r_arr, _ = precision_recall_curve(y_true[mask], y_score[mask])
        ax.plot(r_arr, p_arr, lw=2, color=color, ls=ls,
                label=f"{label} (n={n}, pos={n_pos})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, FIG_PR_STRATIFIED)

    # 2) Stratified P/R/F1 vs threshold (one subplot per usable bucket) ------
    plottable = []
    for label, mask, color, _ls in buckets:
        n_pos = int(y_true[mask].sum())
        if n_pos == 0:
            print(f"  Threshold sweep: skipping bucket '{label}' "
                  f"(no positives → precision undefined)")
            continue
        plottable.append((label, mask, color))

    if plottable:
        fig, axes = plt.subplots(1, len(plottable), figsize=(4.5 * len(plottable), 4),
                                  sharey=True)
        if len(plottable) == 1:
            axes = [axes]
        thresholds = np.linspace(0.0, 1.0, 101)
        for ax, (label, mask, color) in zip(axes, plottable):
            yt = y_true[mask]; ys = y_score[mask]
            n = int(mask.sum()); n_pos = int(yt.sum())
            P, R, F = [], [], []
            for t in thresholds:
                pred = (ys >= t).astype(int)
                tp = int(((yt == 1) & (pred == 1)).sum())
                fp = int(((yt == 0) & (pred == 1)).sum())
                fn = int(((yt == 1) & (pred == 0)).sum())
                p = tp / (tp + fp) if (tp + fp) else 0.0
                r = tp / (tp + fn) if (tp + fn) else 0.0
                f = 2 * p * r / (p + r) if (p + r) else 0.0
                P.append(p); R.append(r); F.append(f)
            # Three metrics distinguished by both colour and line style
            ax.plot(thresholds, P, lw=2, ls="-",  color="#1f77b4", label="Precision")
            ax.plot(thresholds, R, lw=2, ls="--", color="#ff7f0e", label="Recall")
            ax.plot(thresholds, F, lw=2, ls="-.", color="#2ca02c", label="F1")
            # Bucket label inside the panel rather than as a chart title
            ax.text(0.04, 0.96, f"{label}\n(n={n}, pos={n_pos})",
                    transform=ax.transAxes, fontsize=10, verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))
            ax.set_xlabel("Decision threshold τ")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", framealpha=0.95)
        axes[0].set_ylabel("Metric value")
        fig.tight_layout()
        _save(fig, FIG_THRESHOLD_STRATIFIED)


# ── 5. Combined decision-behaviour figure (one figure, three panels) ──────────

def plot_decision_behavior(curve: pd.DataFrame, ops: pd.DataFrame,
                            cost_df: pd.DataFrame, scoring: dict) -> None:
    """Combined paper figure: threshold sensitivity, cost-sensitive shift,
    label-confidence stratification (PR), and the threshold sweep stratified
    by confidence — one figure, four panels in a 2×2 grid.

    Designed to drop directly into the LaTeX placeholder labelled
    `fig:decision-behavior`. Sized for `\\includegraphics[width=\\linewidth]`
    in single-column LNCS layout (2×2 reads more comfortably than 1×4).
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # ── (a) Threshold sensitivity: P / R / F1 vs threshold ──────────────────
    ax = axes[0, 0]
    ax.plot(curve["threshold"], curve["precision"], lw=1.8, ls="-",
            color="#1f77b4", label="Precision")
    ax.plot(curve["threshold"], curve["recall"], lw=1.8, ls="--",
            color="#ff7f0e", label="Recall")
    ax.plot(curve["threshold"], curve["f1"], lw=1.8, ls="-.",
            color="#2ca02c", label="F1")
    f1_op = ops[ops["name"] == "f1_optimal"]
    if len(f1_op):
        t = float(f1_op.iloc[0]["threshold"])
        ax.axvline(t, ls=":", color="black", alpha=0.6, lw=1.0)
    ax.set_xlabel("Decision threshold τ"); ax.set_ylabel("Metric value")
    # Clip the x-axis: precision becomes very noisy beyond τ ≈ 0.7 because
    # only a handful of commits are predicted positive at high thresholds.
    ax.set_xlim(0, 0.7); ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9, framealpha=0.95)
    ax.set_title("(a) Threshold sensitivity", loc="left",
                 fontsize=11, fontweight="bold", pad=8)

    # ── (b) Cost-sensitive shift: cost vs threshold for 3 ratios ────────────
    ax = axes[0, 1]
    ratio_styles = [("1:1", "-"), ("5:1", "--"), ("10:1", "-.")]
    for ratio, ls in ratio_styles:
        sub = cost_df[cost_df["ratio"] == ratio].sort_values("threshold")
        cost_norm = sub["cost"].values / sub["cost"].max()
        ax.plot(sub["threshold"], cost_norm, lw=1.8, ls=ls, color="black",
                label=f"$C_{{FN}}{{:}}C_{{FP}} = {ratio}$")
        op_name = f"cost_optimal_{ratio}"
        match = ops[ops["name"] == op_name]
        if len(match):
            t_opt = float(match.iloc[0]["threshold"])
            # Find the cost at the matching threshold in `sub` (closest row)
            idx = (sub["threshold"] - t_opt).abs().idxmin()
            cost_at_opt = sub.loc[idx, "cost"] / sub["cost"].max()
            ax.scatter([t_opt], [cost_at_opt], s=70, color="white",
                       edgecolors="black", linewidths=1.2, zorder=5)
    ax.set_xlabel("Decision threshold τ"); ax.set_ylabel("Total cost (normalised)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.set_title("(b) Cost-sensitive shift", loc="left",
                 fontsize=11, fontweight="bold", pad=8)

    # ── (c) Label-confidence effects: PR curves by LLM-judge confidence ─────
    ax = axes[1, 0]
    y_score = scoring["y_score"]; y_true = scoring["y_true"]; conf = scoring["conf"]
    buckets = [
        ("mid (0.85–0.95)", (conf >= 0.85) & (conf < 0.95), "#ff7f0e", "--"),
        ("high (≥0.95)",    conf >= 0.95,                   "#2ca02c", "-"),
    ]
    for label, mask, color, ls in buckets:
        n = int(mask.sum()); n_pos = int(y_true[mask].sum())
        if n_pos == 0:
            continue
        p_arr, r_arr, _ = precision_recall_curve(y_true[mask], y_score[mask])
        ax.plot(r_arr, p_arr, lw=1.8, ls=ls, color=color,
                label=f"{label} (n={n})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
    ax.set_title("(c) Label-confidence effects (PR)", loc="left",
                 fontsize=11, fontweight="bold", pad=8)

    # ── (d) Operating-point comparison: raw confusion-matrix counts ─────────
    # Three thresholds (default 0.5, F1-optimal, cost-optimal at 10:1) shown
    # as grouped bars across the three "interesting" outcomes (TP, FP, FN).
    # TN is omitted: it dominates the y-scale (3500–4000) and is the same
    # qualitative story as the other three (more flagging → fewer TN).
    ax = axes[1, 1]
    # Each operating point: (label, csv-name, fill colour, hatch pattern).
    # Hatch makes the bars distinguishable in greyscale printing.
    points = [
        ("Default (τ=0.5)",   "default_0.5",       "#7f7f7f", ""),
        ("F1-optimal",         "f1_optimal",        "#1f77b4", "xxx"),
        ("Cost-optimal 10:1",  "cost_optimal_10:1", "#d62728", "///"),
    ]
    categories = ["TP", "FP", "FN"]
    n_groups   = len(categories)
    bar_width  = 0.25
    x = np.arange(n_groups)

    for i, (display, op_name, color, hatch) in enumerate(points):
        match = ops[ops["name"] == op_name]
        if len(match) == 0:
            continue
        row = match.iloc[0]
        values = [int(row["tp"]), int(row["fp"]), int(row["fn"])]
        offset = (i - 1) * bar_width
        bars = ax.bar(x + offset, values, bar_width,
                      color=color, edgecolor="black", linewidth=0.8,
                      hatch=hatch, label=display)
        # annotate each bar with its count
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v + 8, str(v),
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x); ax.set_xticklabels(categories)
    ax.set_xlabel("Outcome"); ax.set_ylabel("Number of commits (test)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(int(ops[ops["name"] == "cost_optimal_10:1"].iloc[0]["fp"]) * 1.18, 100))
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.set_title("(d) Operating-point comparison", loc="left",
                 fontsize=11, fontweight="bold", pad=8)

    fig.tight_layout()
    _save(fig, FIG_DECISION_BEHAVIOR)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading curve data...")
    curve   = pd.read_csv(PR_CURVE)
    ops     = pd.read_csv(OPS_CSV)
    cost_df = pd.read_csv(COST_CSV)

    print("\nGenerating plots:")
    plot_pr_curve(curve, ops)
    plot_threshold_sweep(curve, ops)
    plot_cost_curve(cost_df, ops)

    # Train once, reuse for both stratified and combined plots.
    scoring = _train_and_score_for_stratification()
    plot_stratified_by_confidence(scoring=scoring)
    plot_decision_behavior(curve, ops, cost_df, scoring)

    print(f"\nAll figures → {FIG_DIR}/")


if __name__ == "__main__":
    main()
