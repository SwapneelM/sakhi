import os
import warnings
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(__file__)
FINAL_FILES_DIR = os.path.join(BASE_DIR, "final_files")

# Consistent color scheme
LANGUAGE_COLORS = {"EN": "#2E86AB", "HI": "#A23B72", "MR": "#F18F01"}
RATER_COLORS = {"expert": "#06A77D", "non_expert": "#D00000"}
METRIC_COLORS = {"mqs": "#7209B7", "semantic": "#F77F00", "linguistic": "#FCBF49"}

LANGUAGE_LABELS = {"EN": "English", "HI": "Hindi", "MR": "Marathi"}

# Model names mapping
MODEL_NAMES = {
    'cohere_command_a': 'Command A',
    'gemini_2_5_flash': 'Gemini 2.5 Flash',
    'gemini_2_5_pro': 'Gemini 2.5 Pro',
    'gpt_5_mini': 'GPT-5 Mini',
    'gpt_4o_mini': 'GPT-4o Mini',
    'llama_3_3_70b': 'Llama 3.3 70B',
    'llama_4_maverick': 'Llama 4 Maverick',
    'aya_expanse': 'Aya Expanse',
    'medgemma_4b': 'MedGemma 4B',
    'medgemma_27b': 'MedGemma 27B'
}

KNOWN_MODELS = [
    'cohere_command_a', 'gemini_2_5_flash', 'gemini_2_5_pro',
    'gpt_5_mini', 'gpt_4o_mini', 'llama_3_3_70b', 'llama_4_maverick',
    'aya_expanse', 'medgemma_4b', 'medgemma_27b'
]

HEATMAP_CMAP = "RdYlGn"
BAR_ALPHA = 0.85

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 10


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def clean_theme_name(theme: str) -> str:
    if theme is None or (isinstance(theme, float) and pd.isna(theme)):
        return ""
    theme_str = str(theme).strip().strip('"').strip("'").strip()
    return theme_str


def infer_rater_type_from_name(name: str) -> str:
    return "expert" if "expert" in name.lower() else "non_expert"


def infer_language_from_col(col: str, default: str = "en") -> str:
    cl = col.lower()
    if cl.endswith("_hi") or "_hindi" in cl or cl.endswith("_hi_linguistic_avg") or cl.endswith("_hi_semantic_avg"):
        return "hi"
    if cl.endswith("_mr") or "_marathi" in cl or cl.endswith("_mr_linguistic_avg") or cl.endswith("_mr_semantic_avg"):
        return "mr"
    return default


def infer_model_from_col(col: str) -> Optional[str]:
    cl = col.lower()
    for model in sorted(KNOWN_MODELS, key=len, reverse=True):
        if cl.startswith(model):
            next_char_idx = len(model)
            if next_char_idx >= len(cl) or cl[next_char_idx] in ['_', '1', '2', '3']:
                return model
    return None


def collect_mqs_long(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    if "theme" not in df.columns:
        return pd.DataFrame()
    rater_type = infer_rater_type_from_name(source_name)
    rows: List[Dict] = []
    for col in df.columns:
        if "judge_mqs" not in col:
            continue
        lang = infer_language_from_col(col)
        model = infer_model_from_col(col)
        vals = pd.to_numeric(df[col], errors="coerce")
        themes = df["theme"].fillna("").astype(str)
        for theme, v in zip(themes, vals):
            if pd.isna(v):
                continue
            cleaned_theme = clean_theme_name(theme)
            if not cleaned_theme:
                continue
            rows.append({
                "theme": cleaned_theme, "language": lang,
                "model": model if model else "unknown",
                "rater_type": rater_type, "metric_type": "mqs",
                "value": float(v), "source": source_name,
            })
    return pd.DataFrame(rows)


def collect_metric_long(df: pd.DataFrame, source_name: str, suffix: str, metric_type: str) -> pd.DataFrame:
    if "theme" not in df.columns:
        return pd.DataFrame()
    rater_type = infer_rater_type_from_name(source_name)
    rows: List[Dict] = []
    for col in df.columns:
        if not col.endswith(suffix):
            continue
        lang = infer_language_from_col(col)
        model = infer_model_from_col(col)
        vals = pd.to_numeric(df[col], errors="coerce")
        themes = df["theme"].fillna("").astype(str)
        for theme, v in zip(themes, vals):
            if pd.isna(v):
                continue
            cleaned_theme = clean_theme_name(theme)
            if not cleaned_theme:
                continue
            rows.append({
                "theme": cleaned_theme, "language": lang,
                "model": model if model else "unknown",
                "rater_type": rater_type, "metric_type": metric_type,
                "value": float(v), "source": source_name,
            })
    return pd.DataFrame(rows)


def load_all_metrics() -> pd.DataFrame:
    if not os.path.isdir(FINAL_FILES_DIR):
        raise FileNotFoundError(f"final_files directory not found: {FINAL_FILES_DIR}")

    csv_files = [f for f in os.listdir(FINAL_FILES_DIR)
                 if (f.endswith(".csv") or f in ["expert_scored.csv", "non_expert_scored.csv"])
                 and not f.startswith(".")]
    if not csv_files:
        raise ValueError(f"No CSV files found in {FINAL_FILES_DIR}")

    long_parts: List[pd.DataFrame] = []
    for fname in csv_files:
        fpath = os.path.join(FINAL_FILES_DIR, fname)
        try:
            df = pd.read_csv(fpath, low_memory=False)
            if df.empty:
                print(f"   Warning: {fname} is empty, skipping")
                continue
        except Exception as e:
            print(f"   Warning: Failed to read {fname}: {e}, skipping")
            continue

        base = os.path.splitext(fname)[0]
        for part in [
            collect_mqs_long(df, base),
            collect_metric_long(df, base, "_linguistic_avg", "linguistic"),
            collect_metric_long(df, base, "_semantic_avg", "semantic"),
        ]:
            if not part.empty:
                long_parts.append(part)

    if not long_parts:
        raise ValueError("No metric data found in any CSV files")

    long_df = pd.concat(long_parts, ignore_index=True)
    lang_map = {"en": "EN", "hi": "HI", "mr": "MR"}
    long_df["language"] = long_df["language"].map(lang_map).fillna(long_df["language"])
    long_df["theme"] = long_df["theme"].apply(clean_theme_name)

    print(f"   Loaded {len(long_df)} metric rows")
    print(f"   Themes: {long_df['theme'].nunique()}")
    print(f"   Languages: {sorted(long_df['language'].unique())}")
    print(f"   Metrics: {sorted(long_df['metric_type'].unique())}")
    if 'model' in long_df.columns:
        models_found = sorted([m for m in long_df['model'].unique() if m != 'unknown'])
        print(f"   Models: {models_found[:10]}{'...' if len(models_found) > 10 else ''}")

    return long_df


# ──────────────────────────────────────────────────────────────────────────────
# Analysis 1 – Theme × Language heatmaps
#   Now ALSO produces one heatmap per language showing all models as columns
# ──────────────────────────────────────────────────────────────────────────────

def plot_theme_language_heatmaps(long_df: pd.DataFrame, out_dir: str) -> None:
    """
    (a) Combined Theme × Language overview heatmap (EN / HI / MR as columns).
    (b) Per-language Theme × Model heatmap so numbers are directly comparable.
    """
    os.makedirs(out_dir, exist_ok=True)

    for metric in ["mqs", "semantic", "linguistic"]:
        sub = long_df[long_df["metric_type"] == metric].copy()
        if sub.empty:
            print(f"   Warning: No {metric} data, skipping")
            continue

        # ── (a) Combined overview ──────────────────────────────────────────
        grouped = sub.groupby(["theme", "language"])["value"].mean().reset_index()
        if not grouped.empty:
            pivot = grouped.pivot(index="theme", columns="language", values="value")
            expected_langs = ["EN", "HI", "MR"]
            for l in expected_langs:
                if l not in pivot.columns:
                    pivot[l] = float("nan")
            pivot = pivot[[l for l in expected_langs if l in pivot.columns]]
            pivot["_mean"] = pivot.mean(axis=1, skipna=True)
            pivot = pivot.sort_values("_mean", ascending=False).drop(columns=["_mean"])
            pivot = pivot[pivot.notna().any(axis=1)]

            if not pivot.empty:
                n_themes = len(pivot)
                fig_h = max(6, min(20, 0.4 * n_themes + 2))
                fig_w = max(8, len(pivot.columns) * 2.5 + 2)
                fig, ax = plt.subplots(figsize=(fig_w, fig_h))
                valid = pivot.values[~pd.isna(pivot.values)]
                vmin = max(0.0, float(valid.min()) - 0.05) if len(valid) else 0.0
                vmax = min(1.0, float(valid.max()) + 0.05) if len(valid) else 1.0
                # Rename columns to full language names for readability
                pivot.columns = [LANGUAGE_LABELS.get(c, c) for c in pivot.columns]
                sns.heatmap(
                    pivot, annot=True, fmt=".3f", cmap=HEATMAP_CMAP,
                    cbar_kws={"label": f"{metric.upper()} Score"},
                    linewidths=0.5, linecolor="white", vmin=vmin, vmax=vmax,
                    ax=ax, annot_kws={"size": 9}, mask=pivot.isna(),
                )
                ax.set_title(
                    f"Theme × Language Overview – {metric.upper()}\n"
                    f"(Mean across all models & raters)",
                    fontsize=13, fontweight="bold", pad=15,
                )
                ax.set_xlabel("Language", fontsize=11, fontweight="bold")
                ax.set_ylabel("Theme", fontsize=11, fontweight="bold")
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
                plt.tight_layout()
                out_path = os.path.join(out_dir, f"theme_language_{metric}_overview.png")
                plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
                plt.close()
                print(f"   ✓ Saved: theme_language_{metric}_overview.png ({n_themes} themes)")

        # ── (b) Per-language Theme × Model breakdown ───────────────────────
        available_langs = sorted(sub["language"].unique())
        for lang in available_langs:
            lang_sub = sub[sub["language"] == lang].copy()
            if lang_sub.empty:
                continue

            # Filter out unknown models
            lang_sub = lang_sub[lang_sub["model"] != "unknown"]
            if lang_sub.empty:
                continue

            grouped_lm = lang_sub.groupby(["theme", "model"])["value"].mean().reset_index()
            if grouped_lm.empty:
                continue

            pivot_lm = grouped_lm.pivot(index="theme", columns="model", values="value")
            if pivot_lm.empty or pivot_lm.isna().all().all():
                continue

            # Sort themes by mean score descending
            pivot_lm["_mean"] = pivot_lm.mean(axis=1, skipna=True)
            pivot_lm = pivot_lm.sort_values("_mean", ascending=False).drop(columns=["_mean"])
            # Sort models by mean score descending
            model_means = pivot_lm.mean(axis=0, skipna=True).sort_values(ascending=False)
            pivot_lm = pivot_lm[model_means.index]
            pivot_lm = pivot_lm[pivot_lm.notna().any(axis=1)]
            pivot_lm = pivot_lm.loc[:, pivot_lm.notna().any(axis=0)]

            if pivot_lm.empty:
                continue

            # Rename model columns
            pivot_lm.columns = [MODEL_NAMES.get(c, c) for c in pivot_lm.columns]

            n_themes = len(pivot_lm)
            n_models = len(pivot_lm.columns)
            fig_h = max(6, min(20, 0.4 * n_themes + 2))
            fig_w = max(10, min(26, 0.9 * n_models + 4))
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))

            valid = pivot_lm.values[~pd.isna(pivot_lm.values)]
            vmin = max(0.0, float(valid.min()) - 0.05) if len(valid) else 0.0
            vmax = min(1.0, float(valid.max()) + 0.05) if len(valid) else 1.0

            sns.heatmap(
                pivot_lm, annot=True, fmt=".3f", cmap=HEATMAP_CMAP,
                cbar_kws={"label": f"{metric.upper()} Score"},
                linewidths=0.5, linecolor="white", vmin=vmin, vmax=vmax,
                ax=ax, annot_kws={"size": 8}, mask=pivot_lm.isna(),
            )
            lang_label = LANGUAGE_LABELS.get(lang, lang)
            ax.set_title(
                f"Theme × Model – {metric.upper()} [{lang_label}]\n"
                f"(Mean across all raters)",
                fontsize=13, fontweight="bold", pad=15,
            )
            ax.set_xlabel("Model", fontsize=11, fontweight="bold")
            ax.set_ylabel("Theme", fontsize=11, fontweight="bold")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            plt.tight_layout()
            fname = f"theme_model_{metric}_{lang}.png"
            plt.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches="tight", facecolor="white")
            plt.close()
            print(f"   ✓ Saved: {fname} ({n_themes} themes × {n_models} models)")


# ──────────────────────────────────────────────────────────────────────────────
# Analysis 2 – Expert vs Non-expert rater disagreement
#   Separate plot per language (already was) PLUS a combined side-by-side
# ──────────────────────────────────────────────────────────────────────────────

def plot_theme_rater_disagreement(long_df: pd.DataFrame, out_dir: str) -> None:
    """
    Per-language bar charts showing Expert vs Non-expert MQS per theme.
    Also produces a combined multi-panel figure for easy cross-language comparison.
    """
    os.makedirs(out_dir, exist_ok=True)

    sub = long_df[long_df["metric_type"] == "mqs"].copy()
    if sub.empty:
        print("   Warning: No MQS data found for rater disagreement plots")
        return

    grp = sub.groupby(["theme", "language", "rater_type"])["value"].mean().reset_index()
    if grp.empty:
        return

    pivot = grp.pivot_table(
        index=["theme", "language"], columns="rater_type", values="value"
    ).reset_index()

    has_expert = "expert" in pivot.columns
    has_non_expert = "non_expert" in pivot.columns
    if not has_expert:
        pivot["expert"] = float("nan")
    if not has_non_expert:
        pivot["non_expert"] = float("nan")

    pivot["abs_diff"] = (pivot["expert"] - pivot["non_expert"]).abs()

    available_langs = sorted(pivot["language"].unique())

    # ── (a) Per-language individual plots ────────────────────────────────
    for lang in available_langs:
        pl = pivot[pivot["language"] == lang].copy()
        pl = pl[pl[["expert", "non_expert"]].notna().any(axis=1)].copy()
        if pl.empty:
            continue

        sort_col = "expert" if (has_expert and pl["expert"].notna().any()) else "non_expert"
        pl_sorted = pl.sort_values(sort_col, ascending=False, na_position="last")

        n_themes = len(pl_sorted)
        fig_h = max(6, min(16, 0.4 * n_themes + 2))
        fig_w = max(10, min(20, n_themes * 0.55 + 4))

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        x = range(n_themes)
        width = 0.35

        if has_expert and pl_sorted["expert"].notna().any():
            ax.bar([i - width / 2 for i in x], pl_sorted["expert"].fillna(0),
                   width=width, label="Expert", color=RATER_COLORS["expert"],
                   alpha=BAR_ALPHA, edgecolor="black", linewidth=0.5)

        if has_non_expert and pl_sorted["non_expert"].notna().any():
            ax.bar([i + width / 2 for i in x], pl_sorted["non_expert"].fillna(0),
                   width=width, label="Non-expert", color=RATER_COLORS["non_expert"],
                   alpha=BAR_ALPHA, edgecolor="black", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(pl_sorted["theme"], rotation=45, ha="right")
        ax.set_ylabel("MQS Score", fontsize=11, fontweight="bold")
        ax.set_xlabel("Theme", fontsize=11, fontweight="bold")
        lang_label = LANGUAGE_LABELS.get(lang, lang)
        ax.set_title(
            f"Expert vs Non-expert MQS by Theme – {lang_label}",
            fontsize=12, fontweight="bold", pad=15,
        )
        ax.legend(loc="upper right", frameon=True, fancybox=True, shadow=True)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_ylim(bottom=0)
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"theme_rater_mqs_{lang}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"   ✓ Saved: theme_rater_mqs_{lang}.png ({n_themes} themes)")

    # ── (b) Combined multi-panel comparison across languages ─────────────
    if len(available_langs) > 1:
        n_langs = len(available_langs)
        all_themes = sorted(pivot["theme"].unique())
        n_themes = len(all_themes)

        fig_w = max(14, n_themes * 0.5 + 4)
        fig, axes = plt.subplots(n_langs, 1, figsize=(fig_w, 5 * n_langs), sharex=False)
        if n_langs == 1:
            axes = [axes]

        for ax, lang in zip(axes, available_langs):
            pl = pivot[pivot["language"] == lang].copy()
            pl = pl[pl[["expert", "non_expert"]].notna().any(axis=1)].copy()
            if pl.empty:
                ax.set_visible(False)
                continue

            sort_col = "expert" if (has_expert and pl["expert"].notna().any()) else "non_expert"
            pl_sorted = pl.sort_values(sort_col, ascending=False, na_position="last")

            x = range(len(pl_sorted))
            width = 0.35
            lang_label = LANGUAGE_LABELS.get(lang, lang)

            if has_expert and pl_sorted["expert"].notna().any():
                ax.bar([i - width / 2 for i in x], pl_sorted["expert"].fillna(0),
                       width=width, label="Expert", color=RATER_COLORS["expert"],
                       alpha=BAR_ALPHA, edgecolor="black", linewidth=0.5)
            if has_non_expert and pl_sorted["non_expert"].notna().any():
                ax.bar([i + width / 2 for i in x], pl_sorted["non_expert"].fillna(0),
                       width=width, label="Non-expert", color=RATER_COLORS["non_expert"],
                       alpha=BAR_ALPHA, edgecolor="black", linewidth=0.5)

            ax.set_xticks(x)
            ax.set_xticklabels(pl_sorted["theme"], rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("MQS Score", fontsize=10, fontweight="bold")
            ax.set_title(f"{lang_label}", fontsize=11, fontweight="bold")
            ax.legend(loc="upper right", frameon=True, fontsize=9)
            ax.grid(axis="y", alpha=0.3, linestyle="--")
            ax.set_ylim(bottom=0)

        fig.suptitle(
            "Expert vs Non-expert MQS by Theme – All Languages",
            fontsize=14, fontweight="bold", y=1.01,
        )
        plt.tight_layout()
        out_path = os.path.join(out_dir, "theme_rater_mqs_ALL_LANGUAGES.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"   ✓ Saved: theme_rater_mqs_ALL_LANGUAGES.png")

    # ── (c) Absolute disagreement heatmap (themes × languages) ───────────
    diff_pivot = pivot.pivot_table(index="theme", columns="language", values="abs_diff")
    if not diff_pivot.empty and not diff_pivot.isna().all().all():
        diff_pivot["_mean"] = diff_pivot.mean(axis=1, skipna=True)
        diff_pivot = diff_pivot.sort_values("_mean", ascending=False).drop(columns=["_mean"])
        diff_pivot.columns = [LANGUAGE_LABELS.get(c, c) for c in diff_pivot.columns]

        n_themes = len(diff_pivot)
        fig_h = max(6, min(20, 0.4 * n_themes + 2))
        fig, ax = plt.subplots(figsize=(max(8, len(diff_pivot.columns) * 2.5 + 2), fig_h))
        sns.heatmap(
            diff_pivot, annot=True, fmt=".3f", cmap="Oranges",
            cbar_kws={"label": "|Expert − Non-expert| MQS"},
            linewidths=0.5, linecolor="white", ax=ax,
            annot_kws={"size": 9}, mask=diff_pivot.isna(),
        )
        ax.set_title(
            "Expert–Non-expert MQS Disagreement by Theme & Language\n"
            "(Higher = larger disagreement)",
            fontsize=13, fontweight="bold", pad=15,
        )
        ax.set_xlabel("Language", fontsize=11, fontweight="bold")
        ax.set_ylabel("Theme", fontsize=11, fontweight="bold")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        plt.tight_layout()
        out_path = os.path.join(out_dir, "theme_rater_disagreement_heatmap.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"   ✓ Saved: theme_rater_disagreement_heatmap.png")


# ──────────────────────────────────────────────────────────────────────────────
# Analysis 3 – Theme × Model heatmaps
#   Separate graphs per language (and per rater type)
# ──────────────────────────────────────────────────────────────────────────────

def _create_theme_model_heatmap(
    pivot: pd.DataFrame,
    metric: str,
    rater_type: Optional[str],
    lang: Optional[str],
    out_dir: str,
) -> None:
    if pivot.empty or pivot.isna().all().all():
        return

    pivot = pivot.copy()
    pivot["_mean"] = pivot.mean(axis=1, skipna=True)
    pivot = pivot.sort_values("_mean", ascending=False).drop(columns=["_mean"])
    model_means = pivot.mean(axis=0, skipna=True).sort_values(ascending=False)
    pivot = pivot[model_means.index]
    pivot = pivot[pivot.notna().any(axis=1)]
    pivot = pivot.loc[:, pivot.notna().any(axis=0)]
    if pivot.empty:
        return

    pivot.columns = [MODEL_NAMES.get(col, col.replace('_', ' ').title()) for col in pivot.columns]

    n_themes = len(pivot)
    n_models = len(pivot.columns)
    fig_h = max(6, min(22, 0.42 * n_themes + 2))
    fig_w = max(10, min(26, 0.95 * n_models + 4))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    valid = pivot.values[~pd.isna(pivot.values)]
    vmin = max(0.0, float(valid.min()) - 0.05) if len(valid) else 0.0
    vmax = min(1.0, float(valid.max()) + 0.05) if len(valid) else 1.0

    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap=HEATMAP_CMAP,
        cbar_kws={"label": f"{metric.upper()} Score"},
        linewidths=0.5, linecolor="white", vmin=vmin, vmax=vmax,
        ax=ax, annot_kws={"size": 7}, mask=pivot.isna(),
    )

    # Build readable title
    rater_label = rater_type.replace("_", " ").title() if rater_type else "All Raters"
    lang_label = LANGUAGE_LABELS.get(lang, lang) if lang else "All Languages"
    ax.set_title(
        f"Theme × Model – {metric.upper()} | {lang_label} | {rater_label}",
        fontsize=13, fontweight="bold", pad=15,
    )
    ax.set_xlabel("Model", fontsize=11, fontweight="bold")
    ax.set_ylabel("Theme", fontsize=11, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()

    # Build filename
    parts = ["theme_model", metric]
    if rater_type:
        parts.append(rater_type)
    if lang:
        parts.append(lang)
    filename = "_".join(parts) + ".png"
    plt.savefig(os.path.join(out_dir, filename), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"   ✓ Saved: {filename} ({n_themes} themes × {n_models} models)")


def plot_theme_model_heatmaps(long_df: pd.DataFrame, out_dir: str) -> None:
    """
    Produces Theme × Model heatmaps for every combination of:
        metric  ∈ {mqs, semantic, linguistic}
        rater   ∈ {overall, expert, non_expert}
        language ∈ {all, EN, HI, MR}

    This gives you per-language separate graphs to compare numbers side by side.
    """
    os.makedirs(out_dir, exist_ok=True)

    if 'model' not in long_df.columns:
        print("   Warning: No model info found, skipping Theme × Model heatmaps")
        return

    ldf = long_df[long_df['model'] != 'unknown'].copy()
    if ldf.empty:
        print("   Warning: No valid model data, skipping Theme × Model heatmaps")
        return

    available_raters = sorted(ldf['rater_type'].unique())
    available_langs = sorted(ldf['language'].unique())
    has_expert = 'expert' in available_raters
    has_non_expert = 'non_expert' in available_raters

    for metric in ["mqs", "semantic", "linguistic"]:
        sub = ldf[ldf["metric_type"] == metric].copy()
        if sub.empty:
            continue

        # Define the (rater_type_filter, rater_label) combinations to plot
        rater_combos = []
        if has_expert and has_non_expert:
            rater_combos.append((None, None))          # overall
        if has_expert:
            rater_combos.append(("expert", "expert"))
        if has_non_expert:
            rater_combos.append(("non_expert", "non_expert"))

        # Define language combos: None = all languages combined, else per-language
        lang_combos = [None] + list(available_langs)

        for rater_filter, rater_label in rater_combos:
            # Filter by rater if needed
            if rater_filter is not None:
                rater_sub = sub[sub['rater_type'] == rater_filter].copy()
            else:
                rater_sub = sub.copy()

            for lang in lang_combos:
                # Filter by language if needed
                if lang is not None:
                    lang_sub = rater_sub[rater_sub['language'] == lang].copy()
                else:
                    lang_sub = rater_sub.copy()

                if lang_sub.empty:
                    continue

                grouped = lang_sub.groupby(["theme", "model"])["value"].mean().reset_index()
                if grouped.empty:
                    continue

                pivot = grouped.pivot(index="theme", columns="model", values="value")
                _create_theme_model_heatmap(pivot, metric, rater_label, lang, out_dir)


# ──────────────────────────────────────────────────────────────────────────────
# Analysis 4 – Model ranking bar charts per language
#   NEW: lets you compare model rankings across EN / HI / MR directly
# ──────────────────────────────────────────────────────────────────────────────

def plot_model_rankings_per_language(long_df: pd.DataFrame, out_dir: str) -> None:
    """
    For each metric, produce:
      (a) A grouped bar chart: models on x-axis, one bar per language.
      (b) A per-language bar chart (one per language) for direct comparison.
    """
    os.makedirs(out_dir, exist_ok=True)

    if 'model' not in long_df.columns:
        return

    ldf = long_df[long_df['model'] != 'unknown'].copy()
    available_langs = sorted(ldf['language'].unique())

    for metric in ["mqs", "semantic", "linguistic"]:
        sub = ldf[ldf["metric_type"] == metric].copy()
        if sub.empty:
            continue

        # Mean per model × language (all raters, all themes)
        grp = sub.groupby(["model", "language"])["value"].mean().reset_index()
        if grp.empty:
            continue

        pivot = grp.pivot(index="model", columns="language", values="value")
        # Sort models by overall mean
        pivot["_mean"] = pivot.mean(axis=1, skipna=True)
        pivot = pivot.sort_values("_mean", ascending=False).drop(columns=["_mean"])
        pivot = pivot[pivot.notna().any(axis=1)]
        if pivot.empty:
            continue

        pivot.index = [MODEL_NAMES.get(m, m) for m in pivot.index]
        if set(available_langs) & {"EN", "HI", "MR"}:
            ordered_cols = [l for l in ["EN", "HI", "MR"] if l in pivot.columns]
            pivot = pivot[ordered_cols + [c for c in pivot.columns if c not in ordered_cols]]

        # ── (a) Grouped bar chart: all languages on one plot ──────────────
        n_models = len(pivot)
        n_langs_plot = len(pivot.columns)
        fig_w = max(12, n_models * 0.9 + 3)
        fig, ax = plt.subplots(figsize=(fig_w, 6))
        x = range(n_models)
        bar_width = 0.8 / max(n_langs_plot, 1)

        for i, lang in enumerate(pivot.columns):
            offset = (i - n_langs_plot / 2 + 0.5) * bar_width
            lang_label = LANGUAGE_LABELS.get(lang, lang)
            ax.bar(
                [xi + offset for xi in x],
                pivot[lang].fillna(0),
                width=bar_width * 0.9,
                label=lang_label,
                color=LANGUAGE_COLORS.get(lang, f"C{i}"),
                alpha=BAR_ALPHA,
                edgecolor="black",
                linewidth=0.5,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=40, ha="right")
        ax.set_ylabel(f"{metric.upper()} Score", fontsize=11, fontweight="bold")
        ax.set_xlabel("Model", fontsize=11, fontweight="bold")
        ax.set_title(
            f"Model Rankings by Language – {metric.upper()}\n"
            f"(Mean across all themes & raters)",
            fontsize=13, fontweight="bold", pad=15,
        )
        ax.legend(title="Language", frameon=True, fancybox=True, shadow=True)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_ylim(bottom=0)
        plt.tight_layout()
        fname = f"model_ranking_{metric}_all_languages.png"
        plt.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"   ✓ Saved: {fname}")

        # ── (b) Individual per-language bar charts ────────────────────────
        for lang in available_langs:
            if lang not in pivot.columns:
                continue
            lang_data = pivot[lang].dropna().sort_values(ascending=False)
            if lang_data.empty:
                continue

            fig_w = max(10, len(lang_data) * 0.9 + 3)
            fig, ax = plt.subplots(figsize=(fig_w, 6))

            bars = ax.bar(
                range(len(lang_data)),
                lang_data.values,
                color=LANGUAGE_COLORS.get(lang, "#2E86AB"),
                alpha=BAR_ALPHA,
                edgecolor="black",
                linewidth=0.5,
            )
            # Add value labels on bars
            for bar_obj, val in zip(bars, lang_data.values):
                ax.text(
                    bar_obj.get_x() + bar_obj.get_width() / 2,
                    bar_obj.get_height() + 0.005,
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold",
                )

            ax.set_xticks(range(len(lang_data)))
            ax.set_xticklabels(lang_data.index, rotation=40, ha="right")
            ax.set_ylabel(f"{metric.upper()} Score", fontsize=11, fontweight="bold")
            ax.set_xlabel("Model", fontsize=11, fontweight="bold")
            lang_label = LANGUAGE_LABELS.get(lang, lang)
            ax.set_title(
                f"Model Rankings – {metric.upper()} [{lang_label}]\n"
                f"(Mean across all themes & raters)",
                fontsize=13, fontweight="bold", pad=15,
            )
            ax.grid(axis="y", alpha=0.3, linestyle="--")
            ax.set_ylim(bottom=0)
            plt.tight_layout()
            fname = f"model_ranking_{metric}_{lang}.png"
            plt.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches="tight", facecolor="white")
            plt.close()
            print(f"   ✓ Saved: {fname}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    try:
        print(">>> Loading metrics from final_files...")
        long_df = load_all_metrics()

        if long_df.empty:
            print("ERROR: No metrics found in final_files; run the scoring pipeline first.")
            return

        graphs_dir = os.path.join(FINAL_FILES_DIR, "graphs")
        os.makedirs(graphs_dir, exist_ok=True)

        print("\n>>> [1] Theme × Language overview heatmaps + per-language Theme × Model breakdown...")
        plot_theme_language_heatmaps(long_df, graphs_dir)

        print("\n>>> [2] Expert vs Non-expert disagreement (per language + combined + disagreement heatmap)...")
        plot_theme_rater_disagreement(long_df, graphs_dir)

        print("\n>>> [3] Theme × Model heatmaps (all language × rater combinations)...")
        plot_theme_model_heatmaps(long_df, graphs_dir)

        print("\n>>> [4] Model rankings per language (grouped + individual)...")
        plot_model_rankings_per_language(long_df, graphs_dir)

        print(f"\n>>> Success! All graphs written to: {graphs_dir}")
        print("\nGraph naming convention:")
        print("  theme_language_<metric>_overview.png      – combined EN/HI/MR heatmap")
        print("  theme_model_<metric>_<lang>.png            – per-language Theme × Model heatmap")
        print("  theme_rater_mqs_<lang>.png                 – Expert vs Non-expert per language")
        print("  theme_rater_mqs_ALL_LANGUAGES.png          – stacked comparison across languages")
        print("  theme_rater_disagreement_heatmap.png       – disagreement magnitude heatmap")
        print("  theme_model_<metric>_<rater>_<lang>.png    – full breakdown heatmaps")
        print("  model_ranking_<metric>_all_languages.png   – grouped bar chart all languages")
        print("  model_ranking_<metric>_<lang>.png          – per-language model ranking")

    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()