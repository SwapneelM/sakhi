"""
Render the main-paper theme figure: theme x language heatmap on the left, three
representative response cards (EN, HI, MR) on the right.

Inspired by the Hivemind-paper layout that juxtaposes a quantitative heatmap
with the actual qualitative free-text it summarises. The reader sees one figure
that says "themes line up like this across the three languages, and here is
what the responses on a hard theme actually look like in each language".

Output:
    overleaf/paper/pictures/fig_theme_with_examples.pdf
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
THEME_CSV = REPO / "release" / "results" / "mqs_per_theme__gpt_4o_mini.csv"
RUNS_DIR = REPO / "runs"
NON_EXPERT_CSV = REPO / "release" / "sakhi_benchmark_non_expert.csv"
OUT_PATH = REPO / "overleaf" / "paper" / "pictures" / "fig_theme_with_examples.pdf"

DARK_TEXT = "#1a1a1a"
GREEN = "#1b9e3f"
RED = "#d12c2c"

THEME_DISPLAY = {
    "Antenatal & Maternal Health Care": "Antenatal",
    "Clinical Procedures & Guidelines": "Clinical",
    "Health Systems, Access & Provider Support": "Health Sys.",
    "Infection Prevention & Hygiene Practices": "Infection",
    "Medication & Vaccination Safety": "Medication",
    "Mental, Emotional & Social Well-being": "Mental",
    "Nutrition, Diet & Supplementation": "Nutrition",
    "Reproductive & Sexual Health (Beyond Pregnancy)": "Reproductive",
    "Risk & Complication Management": "Risk Mgmt.",
    "Symptom Interpretation & Danger Sign Recognition": "Symptoms",
}

LANG_DISPLAY = {"en": "English", "hi": "Hindi", "mr": "Marathi"}

EXAMPLE_MODEL_ALIAS = "claude_haiku_4_5"
EXAMPLE_MODEL_DISPLAY = "Claude Haiku 4.5"
EXAMPLE_THEME_HARD = "Risk & Complication Management"
# Validated 2026-05-06: this q_idx and per-language run choice gives valid
# judge MQS in all three languages (no error) and shows the EN-good /
# HI-mediocre / MR-poor cross-lingual quality drop the figure is illustrating.
EXAMPLE_Q_IDX = 156
EXAMPLE_RUNS = {"en": 3, "hi": 3, "mr": 1}


def find_devanagari_font():
    candidates = [
        "Kohinoor Devanagari",
        "Devanagari MT",
        "ITF Devanagari",
        "Mukta",
        "Annapurna SIL",
        "Lohit Devanagari",
        "Noto Sans Devanagari",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for c in candidates:
        if c in available:
            return c
    return None


def shorten_text(s: str, max_chars: int = 480) -> str:
    s = " ".join(s.split())
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 1].rstrip() + "..."


def wrap_text(s: str, width: int) -> str:
    import textwrap
    s = " ".join(s.split())
    lines = []
    for raw_line in s.split("\n"):
        if not raw_line.strip():
            lines.append("")
            continue
        if " " in raw_line:
            lines.extend(textwrap.wrap(raw_line, width=width) or [""])
        else:
            for i in range(0, len(raw_line), width):
                lines.append(raw_line[i : i + width])
    return "\n".join(lines)


def load_theme_lang_matrix(arm: str = "non_expert") -> tuple[pd.DataFrame, list[str]]:
    """Return a 10x3 (theme x lang) matrix of cross-model mean MQS for the chosen arm."""
    df = pd.read_csv(THEME_CSV)
    df = df[df["dataset"] == arm].copy()
    # Average MQS across models within (theme, lang)
    pivoted = (
        df.groupby(["theme", "lang"])["mqs_mean"]
        .mean()
        .unstack("lang")
        .reindex(columns=["en", "hi", "mr"])
    )
    # Order themes by easiest -> hardest using mean across the three languages
    ordered = pivoted.assign(_mean=pivoted.mean(axis=1)).sort_values("_mean", ascending=False)
    ordered = ordered.drop(columns="_mean")
    # Fold theme display labels
    ordered.index = [THEME_DISPLAY.get(t, t) for t in ordered.index]
    return ordered, list(ordered.index)


def load_example_responses(theme: str, model_alias: str = EXAMPLE_MODEL_ALIAS):
    """Return question + responses + actual MQS for a hand-validated question on
    the given hard theme. We hardcode the q_idx and per-language run choice so
    the example shown matches a valid (non-error) judge verdict and the
    response-level quality matches the per-language MQS pill on the figure.
    """
    bench = pd.read_csv(NON_EXPERT_CSV)
    if EXAMPLE_Q_IDX not in bench.index:
        raise RuntimeError(f"q_idx={EXAMPLE_Q_IDX} not in benchmark CSV")
    if str(bench.loc[EXAMPLE_Q_IDX, "theme"]) != theme:
        raise RuntimeError(
            f"q_idx={EXAMPLE_Q_IDX} is theme {bench.loc[EXAMPLE_Q_IDX,'theme']!r}, expected {theme!r}"
        )

    response_text = {}
    response_mqs = {}
    for lang, run_target in EXAMPLE_RUNS.items():
        # Generation: pull the response text for the chosen run.
        gen_path = RUNS_DIR / f"gen__{model_alias}__non_expert__{lang}.jsonl"
        with gen_path.open() as f:
            for line in f:
                r = json.loads(line)
                if int(r.get("q_idx", -1)) == EXAMPLE_Q_IDX and int(r.get("run", -1)) == run_target:
                    response_text[lang] = (r.get("response") or "").strip()
                    break
        if lang not in response_text:
            raise RuntimeError(f"no gen response for q_idx={EXAMPLE_Q_IDX} run={run_target} lang={lang}")

        # Judge: pull the MQS for the same (q_idx, run) and confirm no error.
        judge_path = RUNS_DIR / f"judge__gpt_4o_mini__{model_alias}__non_expert__{lang}.jsonl"
        with judge_path.open() as f:
            for line in f:
                r = json.loads(line)
                if int(r.get("q_idx", -1)) == EXAMPLE_Q_IDX and int(r.get("run", -1)) == run_target:
                    if r.get("error"):
                        raise RuntimeError(
                            f"judge error for q_idx={EXAMPLE_Q_IDX} run={run_target} lang={lang}: {r['error']!s}"
                        )
                    if r.get("mqs") is None:
                        raise RuntimeError(
                            f"judge mqs is None for q_idx={EXAMPLE_Q_IDX} run={run_target} lang={lang}"
                        )
                    response_mqs[lang] = float(r["mqs"])
                    break
        if lang not in response_mqs:
            raise RuntimeError(
                f"no judge record for q_idx={EXAMPLE_Q_IDX} run={run_target} lang={lang}"
            )

    return {
        "question_en": str(bench.loc[EXAMPLE_Q_IDX, "question_en"]),
        "question_hi": str(bench.loc[EXAMPLE_Q_IDX, "question_hi"]),
        "question_mr": str(bench.loc[EXAMPLE_Q_IDX, "question_mr"]),
        "response_en": response_text["en"],
        "response_hi": response_text["hi"],
        "response_mr": response_text["mr"],
        "mqs_en": response_mqs["en"],
        "mqs_hi": response_mqs["hi"],
        "mqs_mr": response_mqs["mr"],
        "run_en": EXAMPLE_RUNS["en"],
        "run_hi": EXAMPLE_RUNS["hi"],
        "run_mr": EXAMPLE_RUNS["mr"],
    }


def main():
    print("loading theme x language matrix ...")
    matrix, theme_order = load_theme_lang_matrix("non_expert")
    print(matrix.round(3))

    print(f"\nloading example responses for theme {EXAMPLE_THEME_HARD!r} ...")
    examples = load_example_responses(EXAMPLE_THEME_HARD)
    print("question_en:", examples["question_en"][:80])

    deva_font = find_devanagari_font()
    deva_props = (
        font_manager.FontProperties(family=deva_font, size=12) if deva_font else None
    )
    if deva_font:
        print(f"  using Devanagari font: {deva_font}")

    plt.rcParams.update({
        "font.size": 13,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 600,
        "figure.dpi": 200,
    })

    fig = plt.figure(figsize=(16.0, 9.5))
    # Outer grid: heatmap (col 0) and right-hand response panel (col 1). The
    # right panel is itself a 4-row sub-grid: 1 header row + 3 card rows. Each
    # card lives in its own axes so the response-text bboxes physically
    # cannot bleed into the next card's MQS pill.
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    outer = GridSpec(
        1, 2,
        width_ratios=[0.85, 1.15],
        wspace=0.22,
        left=0.06, right=0.98, top=0.92, bottom=0.05,
    )
    gs = outer  # alias kept so the existing heatmap code reads the same
    right_grid = GridSpecFromSubplotSpec(
        4, 1,
        subplot_spec=outer[0, 1],
        height_ratios=[0.45, 1.0, 1.0, 1.0],
        hspace=0.30,
    )

    # ----- LEFT: heatmap -----
    ax_heat = fig.add_subplot(gs[0, 0])
    data = matrix.values  # rows=theme, cols=lang
    cmap = plt.get_cmap("RdYlGn")
    im = ax_heat.imshow(data, cmap=cmap, vmin=0.0, vmax=0.55, aspect="auto")
    ax_heat.set_xticks(range(len(matrix.columns)))
    ax_heat.set_xticklabels(
        [LANG_DISPLAY[c] for c in matrix.columns], fontsize=14, weight="bold"
    )
    ax_heat.set_yticks(range(len(matrix.index)))
    ax_heat.set_yticklabels(matrix.index, fontsize=13)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            # Always-black numerals: RdYlGn at the values we plot here ranges
            # from light-red through yellow to dark green, all of which give
            # readable contrast against black text. White text only worked on
            # the dark-red end and looked broken in the yellow/orange band.
            ax_heat.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                fontsize=14, weight="bold", color="black",
            )
    # Heatmap title removed: the suptitle already names the figure and the
    # caption carries the "avg across 13 models, non-expert arm, GPT-4o-mini
    # judge" context, so a sub-title here only collides with the suptitle.
    # Colour bar without crowding the heatmap.
    cbar = fig.colorbar(im, ax=ax_heat, shrink=0.88, pad=0.02)
    cbar.set_label("MQS  (red = poor, green = good)", fontsize=12)
    cbar.ax.tick_params(labelsize=11)
    # Highlight the hardest theme row (the one we are showing examples for).
    hard_label = THEME_DISPLAY.get(EXAMPLE_THEME_HARD, EXAMPLE_THEME_HARD)
    if hard_label in matrix.index:
        hard_row_idx = matrix.index.get_loc(hard_label)
        rect = mpatches.Rectangle(
            (-0.5, hard_row_idx - 0.5), len(matrix.columns), 1,
            linewidth=2.6, edgecolor="black", facecolor="none", zorder=4,
        )
        ax_heat.add_patch(rect)

    # ----- RIGHT: header + three card axes (one per language) -----
    # Header axes: theme + question (no card box; just running text).
    ax_header = fig.add_subplot(right_grid[0, 0])
    ax_header.set_xlim(0, 1); ax_header.set_ylim(0, 1); ax_header.axis("off")
    q_short = shorten_text(examples["question_en"], 130)
    header_line = (
        f"Same question, three languages on a hard theme  "
        f"({hard_label}, {EXAMPLE_MODEL_DISPLAY})"
    )
    sub_line = f"“{q_short}”"
    ax_header.text(
        0.0, 0.95, header_line,
        fontsize=14, weight="bold", color=DARK_TEXT,
        ha="left", va="top",
    )
    ax_header.text(
        0.0, 0.45, sub_line,
        fontsize=12, color=DARK_TEXT, style="italic",
        ha="left", va="top",
    )

    # Three card axes, one per language. Each card has its own bbox; matplotlib
    # respects subplot boundaries so a long English response cannot overflow
    # into the Hindi MQS pill.
    card_specs = [
        ("en", examples["response_en"], "English", 76, None,
         examples["mqs_en"], examples["run_en"]),
        ("hi", examples["response_hi"], "Hindi (Devanagari)", 64, deva_props,
         examples["mqs_hi"], examples["run_hi"]),
        ("mr", examples["response_mr"], "Marathi (Devanagari)", 64, deva_props,
         examples["mqs_mr"], examples["run_mr"]),
    ]
    for ci, (lang, body, label, wrap_w, fp, mqs_val, run_n) in enumerate(card_specs):
        ax_card = fig.add_subplot(right_grid[ci + 1, 0])
        ax_card.set_xlim(0, 1); ax_card.set_ylim(0, 1); ax_card.axis("off")
        # MQS pill at the top-left of the card axes. The pill is coloured
        # on the same scale as the heatmap, with a hard black border and
        # always-black text so the value reads cleanly on every shade
        # (RdYlGn maps the 0.15-0.30 range to light yellow/orange where
        # white text would disappear).
        swatch_colour = cmap((mqs_val - 0.0) / (0.55 - 0.0))
        ax_card.add_patch(
            mpatches.Rectangle(
                (0.0, 0.85),
                0.115, 0.13,
                linewidth=1.6,
                edgecolor="black",
                facecolor=swatch_colour,
                transform=ax_card.transAxes,
                zorder=5,
            )
        )
        ax_card.text(
            0.0575, 0.915, f"MQS {mqs_val:.2f}",
            fontsize=12, weight="bold",
            color="black",
            ha="center", va="center", zorder=6,
        )
        ax_card.text(
            0.13, 0.915, f"{label}  ·  run {run_n}",
            fontsize=13, weight="bold", color=DARK_TEXT,
            ha="left", va="center",
        )
        # Response body card.
        wrapped = wrap_text(shorten_text(body, 560), wrap_w)
        kwargs = dict(
            fontsize=11.5, color=DARK_TEXT, ha="left", va="top",
            linespacing=1.40,
            bbox=dict(boxstyle="round,pad=0.65", linewidth=1.3,
                      edgecolor="#666", facecolor="white"),
        )
        if fp is not None:
            kwargs["fontproperties"] = fp
        ax_card.text(0.0, 0.78, wrapped, **kwargs)

    fig.suptitle(
        "Theme-by-language difficulty (avg across 13 models, non-expert arm, GPT-4o-mini judge), "
        "with what one model's response actually looks like in each language",
        fontsize=15, weight="bold", y=0.985,
    )

    plt.savefig(OUT_PATH, dpi=600, bbox_inches="tight", pad_inches=0.30)
    print(f"\nwrote {OUT_PATH.relative_to(REPO)}")


if __name__ == "__main__":
    main()
