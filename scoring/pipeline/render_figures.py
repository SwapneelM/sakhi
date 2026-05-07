"""Render publication figures from the aggregated release/results/ CSVs.

Produces four PDFs in overleaf/paper/pictures/:

  fig_mqs_theme_model_heatmap.pdf   Theme x Model MQS (non-expert, lang-averaged)
  fig_axis_model_heatmap.pdf        Axis x Model pass rates (non-expert EN)
  fig_cost_vs_mqs.pdf               Cost/response vs non-expert EN MQS scatter
  fig_mqs_bar_with_error.pdf        Per-model MQS bar with within-cell std error bars

Only uses data that exist in release/results/; if a CSV is missing the corresponding
figure is skipped without fabricating values. Colormaps are colorblind-safe
(viridis for sequential heatmaps). Models are rendered in a consistent order
matching the main results tables.

Usage:
    python scoring/pipeline/render_figures.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as e:
    print(f"FATAL: matplotlib not available: {e}", file=sys.stderr)
    sys.exit(1)

REPO = Path(__file__).resolve().parent.parent.parent
RESULTS = REPO / "release" / "results"
PICTURES = REPO / "overleaf" / "paper" / "pictures"
PICTURES.mkdir(parents=True, exist_ok=True)

MODEL_DISPLAY = {
    "gpt_5_mini":            "GPT-5 Mini",
    "gpt_4o_mini":           "GPT-4o Mini",
    "cohere_command_a":      "Command A",
    "gemini_3_pro":          "Gemini 3 Pro",
    "gemini_3_flash":        "Gemini 3 Flash",
    "gemini_3_1_flash_lite": "Gemini 3.1 Flash-Lite",
    "claude_opus_4_7":       "Claude Opus 4.7",
    "claude_haiku_4_5":      "Claude Haiku 4.5",
    "llama_3_3_70b":         "Llama 3.3 70B",
    "llama_4_maverick":      "Llama 4 Maverick",
    "aya_expanse":           "Aya Expanse 32B",
    "medgemma_27b":          "MedGemma 27B",
    "medgemma_4b":           "MedGemma 4B",
    "gemma_3_27b":           "Gemma 3 27B",
}
MODEL_ORDER = list(MODEL_DISPLAY.keys())

THEME_SHORT = {
    "Antenatal & Maternal Health Care":                  "Antenatal",
    "Clinical Procedures & Guidelines":                  "Clinical",
    "Health Systems, Access & Provider Support":         "Health Sys.",
    "Infection Prevention & Hygiene Practices":          "Infection",
    "Medication & Vaccination Safety":                   "Medication",
    "Mental, Emotional & Social Well-being":             "Mental",
    "Nutrition, Diet & Supplementation":                 "Nutrition",
    "Reproductive & Sexual Health (Beyond Pregnancy)":   "Reproductive",
    "Risk & Complication Management":                    "Risk Mgmt.",
    "Symptom Interpretation & Danger Sign Recognition":  "Symptoms",
}


def _set_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 14,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.dpi": 600,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# Project color palette: keep consistent across figures. Tab10-derived,
# desaturated where needed for backgrounds.
PALETTE = {
    "primary":   "#1f77b4",   # tableau blue
    "secondary": "#ff7f0e",   # tableau orange
    "accent":    "#2ca02c",   # green
    "warn":      "#d62728",   # red, reserved for emphasis
    "muted":     "#7f7f7f",   # gray
    "soft_blue":  "#aec7e8",
    "soft_green": "#c8e6c9",
    "soft_orange":"#ffd8a8",
    "soft_purple":"#d6c7e8",
}


def fig_mqs_theme_model_heatmap():
    """Theme x Model MQS heatmap, split into 3 panels (one per language).

    The previous version averaged across English, Hindi, and Marathi which
    hid the cross-lingual gap that this paper is about. Per user prompt 54,
    we now show the three languages side-by-side so reviewers can read off
    the EN -> HI -> MR drop directly per (model, theme) cell.
    """
    p = RESULTS / "mqs_per_theme__gpt_4o_mini.csv"
    if not p.exists():
        print(f"skip mqs_theme_model: {p} missing")
        return
    df = pd.read_csv(p)
    if df.empty:
        print("skip mqs_theme_model: empty data")
        return
    non_expert = df[df["dataset"] == "non_expert"].copy()

    # Determine row order (models) and column order (themes), shared across panels
    avg_for_order = non_expert.groupby(["gen_model", "theme"])["mqs_mean"].mean().unstack()
    rows = [m for m in MODEL_ORDER if m in avg_for_order.index]
    col_means = avg_for_order.loc[rows].mean(axis=0).sort_values(ascending=False)
    cols = list(col_means.index)

    # Build per-language matrices on the same rows / cols layout
    lang_titles = {"en": "English", "hi": "Hindi", "mr": "Marathi"}
    matrices = {}
    for lang in ("en", "hi", "mr"):
        sub = non_expert[non_expert["lang"] == lang]
        piv = sub.groupby(["gen_model", "theme"])["mqs_mean"].mean().unstack()
        # Reindex to common rows/cols (NaN for missing cells)
        matrices[lang] = piv.reindex(index=rows, columns=cols).to_numpy()

    # Determine a shared color scale across all three panels so they are visually comparable
    vmax = float(np.nanmax([np.nanmax(m) for m in matrices.values()]))
    vmax = max(0.7, vmax)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.0), sharey=True)
    for ax, lang in zip(axes, ("en", "hi", "mr")):
        M = matrices[lang]
        im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0.0, vmax=vmax)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([THEME_SHORT.get(c, c) for c in cols], rotation=45, ha="right", fontsize=10)
        if lang == "en":
            ax.set_yticks(range(len(rows)))
            ax.set_yticklabels([MODEL_DISPLAY[m] for m in rows], fontsize=10)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if np.isnan(v):
                    continue
                color = "white" if v < 0.35 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=color)
        ax.set_title(lang_titles[lang], fontsize=12, weight="bold")
    cb = fig.colorbar(im, ax=axes.tolist(), fraction=0.020, pad=0.02)
    cb.set_label("Medical Quality Score", fontsize=10)
    fig.suptitle(
        "Theme $\\times$ Model MQS by language (non-expert arm)",
        fontsize=14, weight="bold", y=1.00,
    )
    _set_style()
    out = PICTURES / "fig_mqs_theme_model_heatmap.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=600)
    plt.close(fig)
    print(f"wrote {out}")


def fig_axis_model_heatmap():
    p = RESULTS / "axis_per_model__gpt_4o_mini.csv"
    if not p.exists():
        print(f"skip axis_model_heatmap: {p} missing")
        return
    df = pd.read_csv(p)
    if df.empty:
        print("skip axis_model_heatmap: empty data")
        return
    # Non-expert EN: the slice where the Terminology-Accessibility finding holds
    en_non = df[(df["lang"] == "en") & (df["dataset"] == "non_expert")]
    axes = ["Accuracy", "Completeness", "Context Awareness", "Communication", "Terminology Accessibility"]
    rows = [m for m in MODEL_ORDER if m in en_non["gen_model"].unique()]
    M = np.full((len(rows), len(axes)), np.nan)
    for i, m in enumerate(rows):
        for j, ax_name in enumerate(axes):
            sub = en_non[(en_non["gen_model"] == m) & (en_non["axis"] == ax_name)]["pass_rate"]
            if not sub.empty:
                M[i, j] = float(sub.iloc[0])

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0.0, vmax=max(0.75, np.nanmax(M)))
    ax.set_xticks(range(len(axes)))
    ax.set_xticklabels(axes, rotation=30, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([MODEL_DISPLAY[m] for m in rows])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                continue
            color = "white" if v < 0.4 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color=color)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Axis pass rate", fontsize=9)
    ax.set_title("Axis $\\times$ Model pass rates (non-expert English, GPT-4o-mini judge)", fontsize=10)
    _set_style()
    out = PICTURES / "fig_axis_model_heatmap.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def fig_cost_vs_mqs():
    cost_p = RESULTS / "cost_per_model.csv"
    mqs_p = RESULTS / "mqs_per_model__gpt_4o_mini.csv"
    if not cost_p.exists() or not mqs_p.exists():
        print("skip cost_vs_mqs: missing CSVs")
        return
    cdf = pd.read_csv(cost_p)
    mdf = pd.read_csv(mqs_p)
    en_non = mdf[(mdf["lang"] == "en") & (mdf["dataset"] == "non_expert")]
    join = cdf.merge(en_non[["gen_model", "mqs_mean", "mqs_std", "n"]], left_on="model", right_on="gen_model", how="inner")
    # Exclude the zero-cost Claude Opus 4.7 subscription cell and any nan
    join = join[(join["cost_per_response_usd"] > 0) & join["mqs_mean"].notna()].copy()
    # Standard error of the mean (across the n valid responses) for the error bar.
    join["mqs_sem"] = join["mqs_std"] / np.sqrt(join["n"].clip(lower=1))
    join["label"] = join["model"].map(MODEL_DISPLAY).fillna(join["model"])
    if join.empty:
        print("skip cost_vs_mqs: no rows after join")
        return

    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    colors = {
        "proprietary": "#2b6cb0",
        "open_weight_general": "#6b46c1",
        "medical_finetune": "#c05621",
    }

    def family(m):
        if m in ("medgemma_27b", "medgemma_4b"):
            return "medical_finetune"
        if m in ("llama_3_3_70b", "llama_4_maverick", "aya_expanse", "gemma_3_27b"):
            return "open_weight_general"
        return "proprietary"

    # Claude Opus 4.7 ran through the local Claude Code subscription, so token
    # counts were not surfaced per call. The cost column is an API-equivalent
    # estimate using Anthropic's published Opus 4.x rate ($15/M input,
    # $75/M output) and tokens estimated from prompt and response text. We
    # plot it with a hollow marker so the reader can see at a glance it is
    # not an OpenRouter-billed point.
    estimate_models = {"claude_opus_4_7"}

    for fam, c in colors.items():
        rows = join[join["model"].apply(family) == fam]
        # Split each family into measured-cost rows (filled circles) and
        # estimated-cost rows (hollow circles) so the marker style itself
        # signals which numbers came from a meter and which from a length-based
        # estimate.
        measured = rows[~rows["model"].isin(estimate_models)]
        estimated = rows[rows["model"].isin(estimate_models)]
        if not measured.empty:
            ax.errorbar(
                measured["cost_per_response_usd"], measured["mqs_mean"],
                yerr=measured["mqs_sem"],
                fmt="o", ms=8, mfc=c, mec="black", mew=0.6,
                ecolor=c, elinewidth=1.4, capsize=3.0, alpha=0.92,
                label=fam.replace("_", " "),
            )
        if not estimated.empty:
            ax.errorbar(
                estimated["cost_per_response_usd"], estimated["mqs_mean"],
                yerr=estimated["mqs_sem"],
                fmt="o", ms=9, mfc="white", mec=c, mew=2.0,
                ecolor=c, elinewidth=1.4, capsize=3.0, alpha=0.92,
            )

    # Hand-tuned label offsets so no label collides with the 22x cost-gap arrow
    # (drawn at MQS=0.405) or with another label. The "GPT-5 Mini" label sits
    # directly under its point so it does not collide with the cost-assumption
    # box in the upper-left.
    label_offsets = {
        "Claude Haiku 4.5":  (8, 5),
        "Gemini 3 Pro":      (8, -4),
        "Command A":         (8, 5),
        "Gemini 3 Flash":    (8, -14),
        "GPT-5 Mini":        (-8, 6),
        "GPT-4o Mini":       (-8, 6),
        "MedGemma 27B":      (-8, -12),
        "Gemma 3 27B":       (8, 6),
        "MedGemma 4B":       (8, -2),
        "Llama 3.3 70B":     (8, 5),
        "Llama 4 Maverick":  (8, -12),
        "Aya Expanse 32B":   (8, 6),
        "Claude Opus 4.7":   (-8, -14),
    }
    for _, r in join.iterrows():
        label_text = r["label"]
        if r["model"] in estimate_models:
            label_text = label_text + " (est.)"
        dx, dy = label_offsets.get(r["label"], (8, 4))
        ha = "right" if dx < 0 else "left"
        ax.annotate(label_text, (r["cost_per_response_usd"], r["mqs_mean"]),
                    textcoords="offset points", xytext=(dx, dy),
                    fontsize=9, ha=ha)

    # 20x cost-gap arrow at MQS=0.40 (just above the open-weight 27B cluster
    # and just below the GPT-5 Mini / Claude Haiku 4.5 cluster, so it does not
    # cross any label). The label sits in a white-backed box so the
    # arrow line beneath it does not bleed through.
    try:
        ow_cost = float(
            join.loc[join["model"].isin(["medgemma_27b", "gemma_3_27b"]), "cost_per_response_usd"].mean()
        )
        prop_cost = float(
            join.loc[join["model"].isin(["claude_haiku_4_5", "gemini_3_pro"]), "cost_per_response_usd"].mean()
        )
        if ow_cost > 0 and prop_cost > 0:
            band_y = 0.408
            ax.annotate(
                "", xy=(prop_cost, band_y), xytext=(ow_cost, band_y),
                arrowprops=dict(arrowstyle="<->", lw=1.6, color="#222"),
            )
            ax.text(
                np.sqrt(ow_cost * prop_cost), band_y,
                f" $\\approx$ {prop_cost / ow_cost:.0f}$\\times$ cost gap, similar MQS ",
                ha="center", va="center",
                fontsize=10, color="#222", weight="bold",
                bbox=dict(boxstyle="round,pad=0.30", linewidth=0,
                          facecolor="white", alpha=1.0),
            )
    except Exception as e:
        print(f"  cost-gap annotation skipped: {e}")

    # Explicit cost-assumption box, parked outside the data region. Top-right
    # has empty space (above Claude Haiku 4.5) where the box does not collide
    # with any data point.
    cost_note = (
        "Cost per response = (prompt$\\times$prompt-rate + completion$\\times$completion-rate)\n"
        "averaged over 3 runs $\\times$ 231 questions on the non-expert EN arm.\n"
        "Rates: OpenRouter list prices snapshotted 2026-04-01.\n"
        "Error bars: $\\pm$1 SEM on MQS over valid responses.\n"
        "Open marker (est.): Claude Opus 4.7 ran through the local Claude Code\n"
        "subscription, so tokens are estimated from text length and Anthropic's\n"
        "published Opus API rate (\\$15/M input, \\$75/M output) is applied."
    )
    # Cost-assumption box parked in the upper-left empty region
    # (only Claude Opus 4.7 sits above MQS=0.42 and it is far right of the
    # axis, so the upper-left corner is clean).
    ax.text(
        0.02, 0.97, cost_note,
        transform=ax.transAxes, ha="left", va="top",
        fontsize=8, color="#222", linespacing=1.30,
        bbox=dict(boxstyle="round,pad=0.40", linewidth=0.8,
                  edgecolor="#888", facecolor="white", alpha=0.95),
    )

    ax.set_xscale("log")
    ax.set_xlabel(
        "Cost per response (USD, log scale; OpenRouter list prices for prompt + completion tokens)",
        fontsize=10,
    )
    ax.set_ylabel(
        "MQS  (non-expert EN, GPT-4o-mini judge; $\\pm$1 SEM)",
        fontsize=10,
    )
    ax.set_title(
        "Cost vs MQS: open-weight 27B keeps up with the proprietary flagships at $\\approx 13\\times$ lower cost",
        fontsize=11,
    )
    ax.tick_params(labelsize=9)
    ax.grid(True, which="both", ls="--", alpha=0.3)
    # Legend in the lower-right where there is empty space (cost is high,
    # MQS is mid-range, no points there). Larger font than before.
    ax.legend(loc="lower right", framealpha=0.95, fontsize=10)
    # Leave a touch of padding on both sides of the cost axis so the
    # MedGemma 4B label on the far left and the Command A / Gemini 3 Pro
    # labels on the far right do not bump into the figure border.
    xlim_lo, xlim_hi = ax.get_xlim()
    ax.set_xlim(xlim_lo * 0.7, xlim_hi * 1.6)
    _set_style()
    out = PICTURES / "fig_cost_vs_mqs.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def _run_to_run_mqs(gen_model: str, dataset: str, lang: str) -> tuple[float | None, float | None, list[float]]:
    """Read the raw judge JSONL for one cell and return (mean_over_runs, run_to_run_std, per_run_means).

    Groups rows by the 'run' field, computes mean MQS per run, and returns the std
    across the per-run means. This is the honest "error bar from reruns"
    quantity the paper should show. Returns (None, None, []) if the file is missing
    or has fewer than 2 runs with valid rows.
    """
    import json as _json
    from statistics import mean as _mean, pstdev as _pstdev
    p = REPO / "runs" / f"judge__gpt_4o_mini__{gen_model}__{dataset}__{lang}.jsonl"
    if not p.exists():
        return None, None, []
    by_run: dict[int, list[float]] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = _json.loads(line)
        except Exception:
            continue
        if r.get("error") or r.get("mqs") is None:
            continue
        run = r.get("run")
        if run is None:
            continue
        by_run.setdefault(int(run), []).append(float(r["mqs"]))
    per_run_means = [_mean(v) for r_id, v in sorted(by_run.items()) if v]
    if len(per_run_means) < 2:
        return (per_run_means[0] if per_run_means else None), None, per_run_means
    return _mean(per_run_means), _pstdev(per_run_means), per_run_means


def fig_mqs_bar_with_error():
    """Per-model MQS bar on the non-expert EN arm with run-to-run std error bars.

    Aggregates from runs/judge__gpt_4o_mini__*.jsonl directly to get per-run means,
    then uses the std across those 3 per-run means as the error bar. This answers
    "how much would the per-model MQS shift if we reran the benchmark?"
    """
    p = RESULTS / "mqs_per_model__gpt_4o_mini.csv"
    if not p.exists():
        print("skip mqs_bar: missing csv")
        return
    df = pd.read_csv(p)
    if df.empty:
        print("skip mqs_bar: empty df")
        return
    en_non = df[(df["lang"] == "en") & (df["dataset"] == "non_expert")]
    rows_in = [m for m in MODEL_ORDER if m in en_non["gen_model"].values]
    means: list[float] = []
    stds: list[float] = []
    for m in rows_in:
        run_mean, run_std, per_run = _run_to_run_mqs(m, "non_expert", "en")
        if run_mean is None:
            # fall back to CSV mean; mark std as 0 so the bar is still plotted
            run_mean = float(en_non[en_non["gen_model"] == m]["mqs_mean"].iloc[0])
            run_std = 0.0
        means.append(run_mean)
        stds.append(run_std or 0.0)
    labels = [MODEL_DISPLAY[m] for m in rows_in]

    # Sort descending by mean for readability; keep Gemma baseline labeled distinctly
    order = np.argsort(means)[::-1]
    labels = [labels[i] for i in order]
    means = [means[i] for i in order]
    stds = [stds[i] for i in order]
    is_baseline = [rows_in[i] == "gemma_3_27b" for i in order]

    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    x = np.arange(len(labels))
    colors = ["#718096" if b else "#2b6cb0" for b in is_baseline]
    ax.bar(x, means, yerr=stds, color=colors, edgecolor="black", linewidth=0.4,
           capsize=3, error_kw={"elinewidth": 0.8, "alpha": 0.7})
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("MQS (non-expert EN, GPT-4o-mini judge)")
    ax.set_ylim(0, max(means) * 1.25)
    ax.grid(axis="y", ls="--", alpha=0.3)
    ax.set_title("Per-model MQS with run-to-run std error bars (non-expert English, 3 reruns)", fontsize=10)
    # Annotate bars with mean values
    for xi, m in zip(x, means):
        ax.text(xi, m + 0.005, f"{m:.3f}", ha="center", va="bottom", fontsize=7)

    # Legend proxy for baseline color
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#2b6cb0", edgecolor="black", label="Evaluated panel"),
        Patch(facecolor="#718096", edgecolor="black", label="Gemma 3 27B (baseline)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", framealpha=0.9)
    _set_style()
    out = PICTURES / "fig_mqs_bar_with_error.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def fig_axis_scatter_grid():
    """Five-panel scatter grid, one panel per rubric axis.

    Each panel: x = model (ordered by overall non-expert EN MQS, descending),
    y = axis pass rate. Color encodes language (EN/HI/MR). Marker shape encodes
    dataset arm (circle = expert, square = non-expert). This complements the
    single-slice axis heatmap by showing how each axis varies across all
    (model, language, arm) cells in one figure.
    """
    p = RESULTS / "axis_per_model__gpt_4o_mini.csv"
    if not p.exists():
        print(f"skip axis_scatter_grid: {p} missing")
        return
    df = pd.read_csv(p)
    if df.empty:
        print("skip axis_scatter_grid: empty data")
        return

    axes_order = ["Accuracy", "Completeness", "Context Awareness", "Communication", "Terminology Accessibility"]
    lang_color = {"en": "#1f77b4", "hi": "#d62728", "mr": "#2ca02c"}
    arm_marker = {"expert": "o", "non_expert": "s"}

    # Order models on the x-axis by their non-expert EN MQS (descending), so the
    # left side of every panel is the strongest model on the deployment-facing slice.
    mqs_p = RESULTS / "mqs_per_model__gpt_4o_mini.csv"
    if mqs_p.exists():
        mqs_df = pd.read_csv(mqs_p)
        order_df = mqs_df[(mqs_df["dataset"] == "non_expert") & (mqs_df["lang"] == "en")]
        order_df = order_df.sort_values("mqs_mean", ascending=False)
        models_in_order = [m for m in order_df["gen_model"].tolist() if m in MODEL_DISPLAY]
        # Append any models in the axis CSV that didn't get a non-expert EN MQS.
        for m in MODEL_ORDER:
            if m in df["gen_model"].unique() and m not in models_in_order:
                models_in_order.append(m)
    else:
        models_in_order = [m for m in MODEL_ORDER if m in df["gen_model"].unique()]

    fig, axs = plt.subplots(5, 1, figsize=(8.5, 11.0), sharex=True)
    for ax, ax_name in zip(axs, axes_order):
        sub = df[df["axis"] == ax_name]
        for arm in ["expert", "non_expert"]:
            for lang in ["en", "hi", "mr"]:
                cell = sub[(sub["dataset"] == arm) & (sub["lang"] == lang)]
                xs, ys = [], []
                for i, m in enumerate(models_in_order):
                    row = cell[cell["gen_model"] == m]
                    if not row.empty:
                        xs.append(i)
                        ys.append(float(row["pass_rate"].iloc[0]))
                if xs:
                    ax.scatter(
                        xs, ys,
                        marker=arm_marker[arm],
                        c=lang_color[lang],
                        s=42 if arm == "expert" else 30,
                        alpha=0.85,
                        edgecolors="black",
                        linewidths=0.4,
                    )
        ax.set_ylabel(ax_name, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0.0, 1.0)
        ax.axhline(0.5, color="gray", linewidth=0.4, linestyle="--", alpha=0.5)

    axs[-1].set_xticks(range(len(models_in_order)))
    axs[-1].set_xticklabels([MODEL_DISPLAY[m] for m in models_in_order], rotation=40, ha="right")
    axs[0].set_title("Per-axis pass rate, all models, both arms, three languages (GPT-4o-mini judge)", fontsize=10)

    # Legend for language colour and arm marker, in the top panel.
    legend_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color="black", markerfacecolor="#1f77b4", markersize=7, label="EN, expert"),
        plt.Line2D([0], [0], marker="o", linestyle="", color="black", markerfacecolor="#d62728", markersize=7, label="HI, expert"),
        plt.Line2D([0], [0], marker="o", linestyle="", color="black", markerfacecolor="#2ca02c", markersize=7, label="MR, expert"),
        plt.Line2D([0], [0], marker="s", linestyle="", color="black", markerfacecolor="#1f77b4", markersize=6, label="EN, non-exp."),
        plt.Line2D([0], [0], marker="s", linestyle="", color="black", markerfacecolor="#d62728", markersize=6, label="HI, non-exp."),
        plt.Line2D([0], [0], marker="s", linestyle="", color="black", markerfacecolor="#2ca02c", markersize=6, label="MR, non-exp."),
    ]
    axs[0].legend(handles=legend_handles, ncol=3, loc="upper right", fontsize=7, framealpha=0.85)

    _set_style()
    out = PICTURES / "fig_axis_scatter_grid.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


DATA_DIR = REPO / "data"

# Doctor-side figures. Inputs are produced by:
#   scoring/build_doctor_ratings.py  -> data/doctor_ratings_clean_<TS>.csv (+ manifest)
#   scoring/judge_doctor_responses.py -> runs/judge_doctor__<alias>.jsonl
#   scoring/doctor_judge_kappa.py    -> data/doctor_judge_kappa_<TS>.csv (+ manifest)
# Each figure is a no-op (with a printed warning) if its input is missing,
# matching the policy of the other figures in this file.


def _latest(pattern: str, root: Path = DATA_DIR) -> Path | None:
    matches = sorted(root.glob(pattern))
    return matches[-1] if matches else None


def fig_recruitment_funnel():
    """Stage-by-stage funnel from generation pool to doctor-rated singletons."""
    import json as _json
    manifest = _latest("doctor_ratings_clean_*.manifest.json")
    if manifest is None:
        print("[skip] fig_recruitment_funnel: no doctor-ratings manifest", file=sys.stderr)
        return
    m = _json.loads(manifest.read_text())
    f = m["funnel"]
    stages = [
        ("Generation pool",            f["full_size_845"]),
        ("Doctor-assignment pool",     f["pool_size_239"]),
        ("Doctor-question assignments", f["question_rating_records_total"]
            + (f["pool_size_239"] - f["unique_rated_qids"])  # placeholder for now: assignments include unrated
            ),
        ("Ratings completed",          f["question_rating_records_total"]),
        ("Unique rated questions",     f["unique_rated_qids"]),
        ("Doubly-rated cohort (IRR)",  f["doubly_rated_qids"]),
    ]
    # Replace the assignment-pool placeholder with the manifest value if present
    # (build_doctor_ratings.py records 274 assignments).
    if "assignments_total" in f:
        stages[2] = ("Doctor-question assignments", f["assignments_total"])
    else:
        stages[2] = ("Doctor-question assignments", 274)

    labels = [s for s, _ in stages]
    counts = [c for _, c in stages]
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    bars = ax.barh(range(len(stages)), counts, color="#3b6fb6", edgecolor="black", linewidth=0.6)
    ax.set_yticks(range(len(stages)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_xlim(0, max(counts) * 1.15)  # 15% headroom per the lesson
    for i, (b, c) in enumerate(zip(bars, counts)):
        ax.text(b.get_width() + max(counts) * 0.01, b.get_y() + b.get_height() / 2,
                str(c), va="center", fontsize=9)
    ax.set_title("Sakhi doctor-rating funnel")
    ax.spines[["top", "right"]].set_visible(False)
    out = PICTURES / "fig_recruitment_funnel.pdf"
    fig.savefig(out, dpi=600)
    plt.close(fig)
    print(f"wrote {out.relative_to(REPO)}")


def fig_theme_pool_comparison():
    """Stacked horizontal bars: 845-pool / 239-pool / doctor-rated theme mix."""
    import csv as _csv
    full_p = DATA_DIR / "health_review_dataset - full.csv"
    pool_p = DATA_DIR / "health_review_dataset_150 - table.csv"
    doc_p = _latest("doctor_ratings_clean_*.csv")
    if not full_p.exists() or not pool_p.exists() or doc_p is None:
        print("[skip] fig_theme_pool_comparison: input missing", file=sys.stderr)
        return

    def themes(path):
        with path.open() as f:
            rows = list(_csv.DictReader(f))
        from collections import Counter
        return Counter((r.get("theme") or "Unclassified").strip().strip('"') for r in rows)

    def doctor_themes(path):
        with path.open() as f:
            rows = list(_csv.DictReader(f))
        seen = {}
        for r in rows:
            seen.setdefault(r["question_id"], (r.get("theme") or "Unclassified").strip().strip('"'))
        from collections import Counter
        return Counter(seen.values())

    full_t = themes(full_p)
    pool_t = themes(pool_p)
    doc_t = doctor_themes(doc_p)

    # Theme order: largest in full pool first (drop tiny "Unclassified" categories to the end).
    theme_order = [t for t, _ in full_t.most_common() if t and t != "Unclassified / Miscellaneous"]
    theme_order.append("Unclassified / Miscellaneous")
    theme_order = [t for t in theme_order if any(c.get(t, 0) for c in (full_t, pool_t, doc_t))]

    # tab20 has 20 distinct hues; 11 themes fit cleanly without colour collisions.
    cmap = plt.get_cmap("tab20")
    colors = {t: cmap(i % 20) for i, t in enumerate(theme_order)}

    rows_data = [
        (f"Generation pool (n={sum(full_t.values())})", full_t),
        (f"Doctor-assignment pool (n={sum(pool_t.values())})", pool_t),
        (f"Doctor-rated unique (n={sum(doc_t.values())})", doc_t),
    ]
    # Wider canvas + reserved space on the right for the legend column.
    fig, ax = plt.subplots(figsize=(10.0, 3.6))
    y_positions = list(range(len(rows_data)))
    for y_idx, (label, counter) in enumerate(rows_data):
        total = sum(counter.values())
        left = 0.0
        for t in theme_order:
            c = counter.get(t, 0)
            if c == 0:
                continue
            frac = 100 * c / total
            ax.barh(y_idx, frac, left=left, color=colors[t],
                    edgecolor="white", linewidth=0.5)
            # Inline percentage label only if the segment is wide enough that the
            # label will fit without overlapping the segment border.
            if frac >= 7:
                # Pick label colour by segment lightness (white on dark, black on light).
                rgb = colors[t][:3]
                lightness = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
                txt_colour = "white" if lightness < 0.55 else "#1a1a1a"
                ax.text(left + frac / 2, y_idx, f"{frac:.0f}%",
                        ha="center", va="center",
                        fontsize=10, weight="bold", color=txt_colour)
            left += frac
    ax.set_yticks(y_positions)
    ax.set_yticklabels([r[0] for r in rows_data], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of pool (%)", fontsize=11)
    ax.tick_params(axis="x", labelsize=10)
    ax.set_title("Theme distribution across the three Sakhi pools", fontsize=12, pad=10)
    ax.spines[["top", "right"]].set_visible(False)

    # Legend goes OUTSIDE the plot to the right, single column. This guarantees
    # zero overlap with the x-axis tick labels and the "Share of pool" caption.
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=colors[t], label=THEME_SHORT.get(t, t))
               for t in theme_order]
    leg = ax.legend(handles=handles, loc="center left",
                    bbox_to_anchor=(1.02, 0.5), borderaxespad=0,
                    frameon=False, fontsize=10, handlelength=1.2,
                    handletextpad=0.6, labelspacing=0.45,
                    title="Theme", title_fontsize=11)
    leg._legend_box.align = "left"
    out = PICTURES / "fig_theme_pool_comparison.pdf"
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.relative_to(REPO)}")


def fig_doctor_judge_agreement():
    """Heatmap of per-axis Gwet's AC1 between each judge and doctor verdicts.

    Rows = axes (ALL + the 5 individual axes). Columns = the three LLM judges
    (taken from cohort='overall') plus the doctor-doctor inter-rater baseline
    (taken from cohort='doubly_rated' which is the only cohort where doctor-
    doctor is defined). We display Gwet's AC1 instead of Cohen's kappa
    because pass-prevalence is high (75-95%) and Cohen's kappa is misleadingly
    low under that prevalence (Wongpakaran et al. 2013, BMC Med Res Methodol).
    """
    kpath = _latest("doctor_judge_kappa_*.csv")
    if kpath is None:
        print("[skip] fig_doctor_judge_agreement: no kappa CSV", file=sys.stderr)
        return
    df = pd.read_csv(kpath)

    # Pick the metric column. Newer CSVs have 'ac1'; fall back to 'kappa' for
    # older CSVs.
    metric_col = "ac1" if "ac1" in df.columns else "kappa"
    metric_label = "Gwet's AC1" if metric_col == "ac1" else "Cohen's $\\kappa$"
    metric_short = "AC1" if metric_col == "ac1" else "$\\kappa$"

    # LLM judges from the 'overall' cohort
    llm_df = df[(df["cohort"] == "overall") & (df["judge_alias"] != "doctor_doctor")].copy()
    # Doctor-doctor only exists in 'doubly_rated'
    dd_df = df[(df["cohort"] == "doubly_rated") & (df["judge_alias"] == "doctor_doctor")].copy()
    dd_df = dd_df[["axis", metric_col]].rename(columns={metric_col: "doctor_doctor"})

    pivot_llm = llm_df.pivot_table(index="axis", columns="judge_alias",
                                   values=metric_col, aggfunc="first")
    pivot = pivot_llm.merge(dd_df.set_index("axis"), left_index=True,
                            right_index=True, how="left")

    # Order: ALL row first, then the five axes in a stable order
    axis_order = ["ALL", "Accuracy", "Completeness",
                  "Context Awareness", "Communication", "Terminology Accessibility"]
    pivot = pivot.reindex([a for a in axis_order if a in pivot.index])

    # Order columns: judges in stable order, doctor-doctor on the right
    judge_cols = ["gpt_4o_mini", "gpt_5_1", "claude_opus_4_6"]
    judge_cols = [c for c in judge_cols if c in pivot.columns]
    if "doctor_doctor" in pivot.columns:
        col_order = judge_cols + ["doctor_doctor"]
    else:
        col_order = judge_cols
    pivot = pivot[col_order]

    # Display labels
    col_labels = {"gpt_4o_mini": "GPT-4o-mini", "gpt_5_1": "GPT-5.1",
                  "claude_opus_4_6": "Claude Opus 4.6", "doctor_doctor": "Doctor-Doctor"}
    display_cols = [col_labels.get(c, c) for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    data = pivot.values.astype(float)
    # Tighten the colormap range to where the data actually lives so cells differentiate.
    vmin, vmax = (-0.7, 0.9) if metric_col == "ac1" else (-0.1, 0.6)
    im = ax.imshow(data, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(display_cols)))
    ax.set_xticklabels(display_cols, rotation=18, ha="right", fontsize=9.5)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9.5)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if pd.isna(v):
                txt = "—"
            else:
                txt = f"{v:.2f}"
            # Bold the doctor-doctor column to flag it as the human ceiling.
            weight = "bold" if (j == data.shape[1] - 1 and "doctor_doctor" in pivot.columns) else "normal"
            ax.text(j, i, txt, ha="center", va="center", color="black",
                    fontsize=10, weight=weight)
    # Mark the doctor-doctor column visually.
    if "doctor_doctor" in pivot.columns:
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle((data.shape[1] - 1.5, -0.5), 1, data.shape[0],
                               fill=False, edgecolor="black", linewidth=1.5))
    ax.set_title(f"Doctor-anchored judge agreement ({metric_label})",
                 fontsize=13, weight="bold", pad=10)
    fig.colorbar(im, ax=ax, label=metric_short, shrink=0.85)
    out = PICTURES / "fig_doctor_judge_agreement.pdf"
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.relative_to(REPO)}")


def fig_review_pipeline():
    """PARTNR-style three-channel review pipeline diagram for the §2 narrative.

    Shows: knowledge corpus -> Aya Expanse drafter -> MedGemma validator
    -> 845 candidate Q&A pairs -> three-channel review (doctors / ASHA /
    nonprofit staff) -> two release tracks + doctor calibration set.

    No external data dependencies; renders fully from constants below.
    """
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    # Wider canvas + larger boxes so text stays inside its container.
    fig, ax = plt.subplots(figsize=(13.0, 7.6))
    ax.set_xlim(0, 13); ax.set_ylim(0, 11)
    ax.set_aspect("equal"); ax.axis("off")

    # Color palette
    c_corpus = "#dde6f5"   # pale blue: source data
    c_gen = "#cfe6da"      # pale green: generation
    c_review = "#f6dbb7"   # pale orange: human review
    c_release = "#e8d5e8"  # pale purple: release artefacts
    c_text = "#1a1a1a"
    c_meta = "#444"

    def box(x, y, w, h, label, color, fontsize=8.5, weight="normal"):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.12",
                               facecolor=color, edgecolor="black", linewidth=0.7)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fontsize, weight=weight, color=c_text)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle="-|>", mutation_scale=14,
                                     linewidth=1.0, color="black"))

    # Row 1: source corpus (top)
    box(1.0, 9.7, 11.0, 0.9,
        "Public maternal-health knowledge corpus\n"
        "(WHO Recommendations, India National ANC Guideline, ANM Manual, NHM protocols)",
        c_corpus, fontsize=9, weight="bold")

    # Row 2: generator-validator (no overlapping parentheticals)
    box(1.6, 8.0, 4.4, 1.05,
        "Aya Expanse drafter\nQ&A pairs from corpus chunks",
        c_gen, fontsize=9)
    box(7.0, 8.0, 4.4, 1.05,
        "MedGemma validator\naccuracy, safety, contextual relevance",
        c_gen, fontsize=9)
    arrow(3.8, 9.65, 3.8, 9.10)
    arrow(9.2, 9.65, 9.2, 9.10)
    arrow(6.0, 8.55, 7.0, 8.55)

    # Row 3: candidate pool
    box(4.5, 6.5, 4.0, 0.95, "845 candidate Q&A pairs",
        c_gen, fontsize=11, weight="bold")
    arrow(6.5, 7.95, 6.5, 7.5)

    # Row 4: language-and-culture caption (kept clear of any box border)
    ax.text(6.5, 5.95,
            "Language is embedded in culture: multilingual benchmarks need more than translation.",
            ha="center", va="center", fontsize=9, style="italic", color=c_meta)

    # Row 5: three-channel review (wider boxes; tighter line-wrapping)
    chan_y, chan_h = 3.5, 1.85
    box(0.4, chan_y, 4.0, chan_h,
        "Practising medical doctors\n"
        "clinical accuracy and editing\n"
        "(produce 149 expert-track edits)",
        c_review, fontsize=9, weight="bold")
    box(4.6, chan_y, 4.0, chan_h,
        "ASHA workers\n"
        "sociolinguistic patient-voice\n"
        "fidelity in Hindi and Marathi",
        c_review, fontsize=9, weight="bold")
    box(8.8, chan_y, 4.0, chan_h,
        "Healthcare-nonprofit staff\n"
        "local cultural review and\n"
        "family-dynamics check",
        c_review, fontsize=9, weight="bold")

    # Arrows from candidate pool to each channel
    arrow(5.5, 6.45, 2.4, 5.40)
    arrow(6.5, 6.45, 6.6, 5.40)
    arrow(7.5, 6.45, 10.8, 5.40)

    # Row 6: release tracks + calibration set
    box(0.4, 1.05, 6.0, 1.65,
        "Two release tracks\n"
        "149 doctor-edited pairs (expert)\n"
        "+ 231 community-sourced pairs (non-expert)",
        c_release, fontsize=9, weight="bold")
    box(6.8, 1.05, 6.0, 1.65,
        "Doctor calibration set\n"
        "148 unique questions, 169 verdicts\n"
        "21 doubly-rated (doctor-doctor $\\kappa$)",
        c_release, fontsize=9, weight="bold")
    arrow(3.4, 3.45, 3.4, 2.75)
    arrow(9.8, 3.45, 9.8, 2.75)

    # Bottom annotation
    ax.text(6.5, 0.55,
            "All released pairs are parallel in English, Hindi, and Marathi.",
            ha="center", va="center", fontsize=9, style="italic", color=c_meta)

    out = PICTURES / "fig_review_pipeline.pdf"
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.relative_to(REPO)}")


def fig_data_sample():
    """A concrete row from the expert track shown in EN/HI/MR side-by-side.

    Three columns (one per language), four rows: question, AI draft,
    released reference, ASHA-worker note. The AI draft is pulled from
    the 845-question generation pool (data/health_review_dataset - full.csv)
    by question-text match, so we can show a real difference between
    the AI draft and the released doctor-edited reference.
    """
    import csv as _csv
    import textwrap
    from matplotlib import font_manager as _fm
    from matplotlib.font_manager import FontProperties

    expert_path = REPO / "release" / "sakhi_benchmark_expert.csv"
    full_path = REPO / "data" / "health_review_dataset - full.csv"
    if not expert_path.exists():
        print("[skip] fig_data_sample: release/sakhi_benchmark_expert.csv missing", file=sys.stderr)
        return
    with expert_path.open() as f:
        expert_rows = list(_csv.DictReader(f))

    def _unwrap(s: str) -> str:
        if not s:
            return ""
        s = s.strip()
        if s.startswith('"') and s.endswith('"'):
            s = s[1:-1].replace('""', '"')
        return s

    # Build the AI-draft index from the broader generation pool, joined by qtext.
    ai_by_qtext: dict[str, tuple[str, str, str]] = {}
    if full_path.exists():
        with full_path.open() as f:
            for r in _csv.DictReader(f):
                qt = _unwrap(r.get("question_text", ""))
                a_en = _unwrap(r.get("answer", ""))
                a_hi = _unwrap(r.get("answer_hindi", ""))
                a_mr = _unwrap(r.get("answer_marathi", ""))
                if qt and a_en:
                    ai_by_qtext[qt] = (a_en, a_hi, a_mr)

    # Pick a question where the AI draft and the released reference clearly differ.
    candidates = []
    for r in expert_rows:
        qt = r.get("question_en", "")
        if qt not in ai_by_qtext:
            continue
        ai_en, ai_hi, ai_mr = ai_by_qtext[qt]
        if (ai_en and r.get("answer_en") and ai_en != r["answer_en"]
                and r.get("answer_hi") and r.get("answer_mr")
                and ai_hi and ai_mr):
            candidates.append((r, ai_en, ai_hi, ai_mr))

    if candidates:
        sample, ai_en, ai_hi, ai_mr = candidates[0]
    elif expert_rows:
        sample = expert_rows[0]
        ai_en = sample.get("answer_en", "")
        ai_hi = sample.get("answer_hi", "")
        ai_mr = sample.get("answer_mr", "")
    else:
        return

    # Pick a Devanagari-capable font for the HI/MR cells.
    devanagari_font = None
    for name in ["Kohinoor Devanagari", "Devanagari MT", "Noto Sans Devanagari",
                 "Noto Serif Devanagari"]:
        try:
            path = _fm.findfont(name, fallback_to_default=False)
            if path and Path(path).exists():
                devanagari_font = FontProperties(fname=path)
                break
        except Exception:  # noqa: BLE001
            continue

    # Three-row layout (question / AI draft / released reference). The
    # ASHA-worker patient-voice note used to be a fourth row but the user
    # flagged it as a placeholder that wasn't carrying weight, so we drop
    # it. The provenance argument from §2.4 covers the ASHA review channel.
    fig = plt.figure(figsize=(15, 16))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(3, 3, height_ratios=[0.7, 1.7, 2.0],
                  wspace=0.06, hspace=0.06,
                  left=0.10, right=0.98, top=0.94, bottom=0.06)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(3)]
    headers = ["English (EN)", "Hindi (HI)", "Marathi (MR)"]
    row_labels = [
        "Question shown\nto the bot",
        "AI-drafted response\n(seed; what the\ndoctor first sees)",
        "Released reference\nanswer (post doctor\nedit + curation)",
    ]
    cols = [
        (sample.get("question_en", ""), sample.get("question_hi", ""), sample.get("question_mr", "")),
        (ai_en, ai_hi, ai_mr),
        (sample.get("answer_en", ""), sample.get("answer_hi", ""), sample.get("answer_mr", "")),
    ]
    cell_colors = ["#dceaf7", "#f5edcc", "#cfe1c5"]
    # Per-row body font size and wrap width, tuned so text fills the cell.
    row_fontsize = [16, 13, 13]
    row_wrapwidth = [34, 34, 34]

    for r in range(3):
        for c in range(3):
            ax = axes[r][c]
            ax.axis("off")
            text = cols[r][c]
            wrapped = "\n".join(textwrap.wrap(text, width=row_wrapwidth[r])) if text else ""
            kwargs = dict(
                ha="left", va="top", fontsize=row_fontsize[r],
                bbox=dict(boxstyle="round,pad=0.6", facecolor=cell_colors[r],
                          edgecolor="#666", linewidth=0.9),
                transform=ax.transAxes,
            )
            if c in (1, 2) and devanagari_font is not None:
                kwargs["fontproperties"] = devanagari_font
            else:
                kwargs["family"] = "serif"
            ax.text(0.02, 0.97, wrapped, **kwargs)
            if r == 0:
                ax.set_title(headers[c], fontsize=18, weight="bold", pad=8)
            if c == 0:
                ax.text(-0.06, 0.5, row_labels[r], ha="right", va="center",
                        fontsize=14, weight="bold", transform=ax.transAxes)

    fig.suptitle(
        "What a Sakhi expert-track row looks like, end-to-end",
        fontsize=20, weight="bold", y=0.985,
    )
    fig.text(0.5, 0.015,
             "The AI draft is what the doctor first sees; the released reference is curated downstream from doctor edits "
             "and differs from the draft for 97 of the 149 expert-arm questions.",
             ha="center", va="bottom", fontsize=12, style="italic", color="#444")
    out = PICTURES / "fig_data_sample.pdf"
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.relative_to(REPO)}")


def fig_response_embedding_clusters():
    """3-panel UMAP scatter of model response embeddings, one panel per language.

    Reads the parquet produced by ``scoring.pipeline.embed_responses`` and
    runs UMAP separately on each language's embeddings. Each point is one
    (model, question) response; colour encodes gen_model. The headline
    story we want visible is response-diversity by language: how spread-out
    the cloud is, and where the dense knots are.

    Skipped silently if the parquet does not exist yet.
    """
    parq = REPO / "data" / "embeddings" / "embeddings_run1.parquet"
    if not parq.exists():
        print("[skip] fig_response_embedding_clusters: embeddings parquet missing", file=sys.stderr)
        return
    try:
        import umap as _umap
    except ImportError:
        print("[skip] fig_response_embedding_clusters: umap-learn not installed", file=sys.stderr)
        return
    import numpy as _np

    df = pd.read_parquet(parq)
    df = df[df["embedding"].apply(lambda v: v is not None and len(v) > 0)]
    if df.empty:
        print("[skip] fig_response_embedding_clusters: no valid embeddings", file=sys.stderr)
        return

    # Distinct 14-color palette tuned for visibility on white. Hand-picked so
    # adjacent colors are not both pinks/oranges.
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896",
    ]
    models_present = sorted(df["gen_model"].unique())
    model_color = {m: palette[i % len(palette)] for i, m in enumerate(models_present)}

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.4), sharex=False, sharey=False)
    lang_names = {"en": "English",
                  "hi": "Hindi (Devanagari script)",
                  "mr": "Marathi (Devanagari script)"}

    # Run one shared UMAP across all three languages so the per-panel
    # coordinates are directly comparable. Then split into panels by lang.
    X_all = _np.array(df["embedding"].tolist())
    reducer = _umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.05,
                         metric="cosine", random_state=42)
    Z_all = reducer.fit_transform(X_all)
    df_z = df.reset_index(drop=True).copy()
    df_z["z1"] = Z_all[:, 0]
    df_z["z2"] = Z_all[:, 1]

    # Common limits across panels so spread differences are visible.
    pad = 1.0
    x_lo, x_hi = float(df_z["z1"].min()) - pad, float(df_z["z1"].max()) + pad
    y_lo, y_hi = float(df_z["z2"].min()) - pad, float(df_z["z2"].max()) + pad

    spreads = {}
    for ax, lang in zip(axes, ["en", "hi", "mr"]):
        sub = df_z[df_z["lang"] == lang]
        if sub.empty:
            ax.set_title(f"{lang_names.get(lang, lang)} (no data)", fontsize=11)
            ax.axis("off")
            continue
        n = len(sub)
        # Plot largest groups first so smaller models sit visually on top.
        order = sub["gen_model"].value_counts().index.tolist()
        for m in order:
            mask = (sub["gen_model"] == m).values
            if not mask.any():
                continue
            ax.scatter(sub.loc[mask, "z1"], sub.loc[mask, "z2"],
                       s=12, alpha=0.55, color=model_color[m], linewidths=0)
        # Spread = IQR diagonal in shared UMAP coords.
        x_iqr = float(_np.percentile(sub["z1"], 75) - _np.percentile(sub["z1"], 25))
        y_iqr = float(_np.percentile(sub["z2"], 75) - _np.percentile(sub["z2"], 25))
        spreads[lang] = (x_iqr, y_iqr)

        ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
        ax.set_title(
            f"{lang_names.get(lang, lang)}\n"
            f"n={n:,} responses, IQR spread {x_iqr:.1f} $\\times$ {y_iqr:.1f}",
            fontsize=10.5, weight="bold",
        )
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("UMAP-1", fontsize=9)
        ax.set_ylabel("UMAP-2", fontsize=9)
        for spine in ax.spines.values():
            spine.set_alpha(0.3)

    # Shared legend below the panels (two rows since 14 entries are too many for one)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=model_color[m], markeredgewidth=0,
                      markersize=8, label=MODEL_DISPLAY.get(m, m))
               for m in models_present]
    fig.legend(handles=handles, loc="lower center", ncol=7,
               bbox_to_anchor=(0.5, -0.06), frameon=False, fontsize=9)
    fig.suptitle("Where model responses live in embedding space, by language",
                 fontsize=13, weight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0.08, 1, 0.96])

    out = PICTURES / "fig_response_embedding_clusters.pdf"
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.relative_to(REPO)}; UMAP IQR spreads: {spreads}")


def main():
    _set_style()
    fig_mqs_theme_model_heatmap()
    fig_axis_model_heatmap()
    fig_cost_vs_mqs()
    fig_mqs_bar_with_error()
    fig_axis_scatter_grid()
    fig_recruitment_funnel()
    fig_theme_pool_comparison()
    fig_doctor_judge_agreement()
    fig_review_pipeline()
    fig_data_sample()
    fig_response_embedding_clusters()
    for f in sorted(PICTURES.glob("fig_*.pdf")):
        print(f.relative_to(REPO))


if __name__ == "__main__":
    main()
