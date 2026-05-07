"""
Render the new headline figure for the Sakhi paper.

Layout (compact horizontal version, prompts 68-69):

    +-------------+ +-------------+ +-------------+
    |  English    | |  Hindi      | |  Marathi    |  scatter row
    |  UMAP       | |  UMAP       | |  UMAP       |
    +-------------+ +-------------+ +-------------+
    +-------------+ +-------------+ +-------------+
    |  good EN    | |  good HI    | |  good MR    |  example row
    |  response   | |  response   | |  response   |  (green outline)
    +-------------+ +-------------+ +-------------+

Each scatter point is one (model, question, run) response, coloured by
its MQS on a red->yellow->green diverging gradient. Below each panel
sits one high-MQS example response (in the language of that panel) in a
green-outlined card.

Output:
    overleaf/paper/pictures/fig_response_embedding_clusters.pdf
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import umap

REPO = Path(__file__).resolve().parents[2]
EMBED_PATH = REPO / "data" / "embeddings" / "embeddings_run1.parquet"
RUNS_DIR = REPO / "runs"
OUT_PATH = REPO / "overleaf" / "paper" / "pictures" / "fig_response_embedding_clusters.pdf"

GREEN = "#1b9e3f"
RED = "#d12c2c"
GRAY = "#888888"
DARK_TEXT = "#1a1a1a"


def find_devanagari_font():
    """Pick a Devanagari-capable font from the system; matplotlib's defaults can't render it."""
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


def load_mqs_lookup() -> dict:
    """Build {(gen_model_alias, dataset, lang, run, q_idx) -> mqs}."""
    out = {}
    for path in RUNS_DIR.glob("judge__gpt_4o_mini__*.jsonl"):
        parts = path.stem.split("__")
        if len(parts) != 5:
            continue
        gen_model, dataset, lang = parts[2], parts[3], parts[4]
        with path.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("error") or r.get("mqs") is None:
                    continue
                key = (gen_model, dataset, lang, int(r.get("run", 1)), int(r["q_idx"]))
                out[key] = float(r["mqs"])
    return out


def shorten_text(s: str, max_chars: int = 240) -> str:
    s = " ".join(s.split())
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 1].rstrip() + "..."


def wrap_text(s: str, width: int) -> str:
    """Hard-wrap a string at roughly `width` characters per line, on word boundaries.

    Devanagari has no spaces between words for some content — fall back to char-wrap
    when no whitespace exists in a long token.
    """
    import textwrap
    s = " ".join(s.split())
    lines = []
    for raw_line in s.split("\n"):
        if not raw_line.strip():
            lines.append("")
            continue
        # Word-wrap if any spaces exist; otherwise hard-wrap by char count.
        if " " in raw_line:
            lines.extend(textwrap.wrap(raw_line, width=width) or [""])
        else:
            for i in range(0, len(raw_line), width):
                lines.append(raw_line[i : i + width])
    return "\n".join(lines)


def pick_high_example(df: pd.DataFrame) -> pd.Series:
    """Return one high-MQS row with a non-trivial response length."""
    sub = df[df["response"].apply(lambda r: 80 < len(r) < 600)].copy()
    if len(sub) < 5:
        sub = df.copy()
    sub = sub.sort_values("mqs")
    return sub.iloc[-(len(sub) // 8) - 1] if len(sub) > 16 else sub.iloc[-1]


def main():
    print("loading embeddings + MQS ...")
    df = pd.read_parquet(EMBED_PATH).copy()
    mqs_lookup = load_mqs_lookup()

    df["mqs"] = df.apply(
        lambda r: mqs_lookup.get((r["gen_model"], r["dataset"], r["lang"], int(r["run"]), int(r["q_idx"]))),
        axis=1,
    )
    before = len(df)
    df = df.dropna(subset=["mqs"]).reset_index(drop=True)
    print(f"  dropped {before - len(df)} rows without MQS; using {len(df)}")

    print("running shared UMAP ...")
    X = np.stack(df["embedding"].values)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    reducer = umap.UMAP(n_neighbors=20, min_dist=0.05, n_components=2,
                        metric="cosine", random_state=42)
    coords = reducer.fit_transform(X)
    df["u1"], df["u2"] = coords[:, 0], coords[:, 1]

    deva_font = find_devanagari_font()
    if deva_font:
        deva_props = font_manager.FontProperties(family=deva_font, size=13)
        print(f"  using Devanagari font: {deva_font}")
    else:
        deva_props = None
        print("  WARNING: no Devanagari font found; HI/MR text may not render")

    plt.rcParams.update({
        "font.size": 14,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 600,
        "figure.dpi": 200,
    })

    spread_strs = {}
    for lang in ["en", "hi", "mr"]:
        sub = df[df["lang"] == lang]
        x = sub["u1"].values
        y = sub["u2"].values
        iqr_x = float(np.percentile(x, 75) - np.percentile(x, 25))
        iqr_y = float(np.percentile(y, 75) - np.percentile(y, 25))
        spread_strs[lang] = f"{iqr_x:.1f} × {iqr_y:.1f}"

    # Shared xlim / ylim so the three panels visually show that Hindi and
    # Marathi occupy a much smaller region of the SAME UMAP space that English
    # spreads across. Without this, each panel auto-zooms to its own data
    # range and the visual contradicts the IQR claim.
    pad_x = 0.05 * (df["u1"].max() - df["u1"].min())
    pad_y = 0.05 * (df["u2"].max() - df["u2"].min())
    shared_xlim = (float(df["u1"].min() - pad_x), float(df["u1"].max() + pad_x))
    shared_ylim = (float(df["u2"].min() - pad_y), float(df["u2"].max() + pad_y))

    # Compact horizontal layout: 3 scatter panels on top, 3 example cards below.
    # Heights tuned so the example cards have enough vertical room to fit a 240-char
    # body wrapped at ~52 chars per line at 13-pt font, with generous breathing
    # room around the green-outlined card. This is the headline figure of the
    # paper, so we trade some pixels for legibility.
    fig = plt.figure(figsize=(17.0, 9.5))
    gs = GridSpec(
        2, 3,
        height_ratios=[2.0, 1.6],
        hspace=0.55, wspace=0.40,
        left=0.05, right=0.98, top=0.88, bottom=0.04,
    )

    cmap = plt.get_cmap("RdYlGn")
    vmin = float(df["mqs"].quantile(0.05))
    vmax = float(df["mqs"].quantile(0.95))
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    lang_titles = {
        "en": "English",
        "hi": "Hindi (Devanagari)",
        "mr": "Marathi (Devanagari)",
    }

    for col, lang in enumerate(["en", "hi", "mr"]):
        ax_scatter = fig.add_subplot(gs[0, col])
        sub = df[df["lang"] == lang].copy()
        order = sub["mqs"].rank(method="first").values
        sub = sub.iloc[np.argsort(order)]
        ax_scatter.scatter(
            sub["u1"], sub["u2"],
            c=sub["mqs"], cmap=cmap, norm=norm,
            s=85, alpha=0.78, edgecolor="none",
        )
        ax_scatter.set_title(
            f"{lang_titles[lang]}  (n={len(sub)})",
            fontsize=17, weight="bold", loc="left", pad=12,
        )
        # The "spread metric should be more visually central" feedback: the IQR
        # box is now sized as a headline annotation overlaid in the upper-right
        # corner with two-line layout (label + value), thicker border, larger
        # font, so the cross-language collapse is the first thing the eye sees.
        ax_scatter.text(
            0.965, 0.955,
            f"IQR\n{spread_strs[lang]}",
            transform=ax_scatter.transAxes,
            ha="right", va="top",
            fontsize=17, weight="bold", color=DARK_TEXT, linespacing=1.05,
            bbox=dict(boxstyle="round,pad=0.55", linewidth=1.6,
                      edgecolor=DARK_TEXT, facecolor="white", alpha=0.94),
        )
        ax_scatter.set_xlabel("UMAP-1", fontsize=13)
        if col == 0:
            ax_scatter.set_ylabel("UMAP-2", fontsize=13)
        ax_scatter.tick_params(axis="both", which="both", labelsize=11)
        ax_scatter.set_xlim(shared_xlim)
        ax_scatter.set_ylim(shared_ylim)

        # Pick one high-MQS exemplar
        high_row = pick_high_example(sub)

        # Bottom card: green-outlined high-MQS example
        ax_text = fig.add_subplot(gs[1, col])
        ax_text.set_xlim(0, 1)
        ax_text.set_ylim(0, 1)
        ax_text.axis("off")

        mqs_val = float(high_row["mqs"])
        # Single shared wrap width across all three languages so the three
        # boxes are visually the same width and don't bump into one another.
        # Truncate response to 240 chars so the box height is also bounded.
        wrap_width = 50
        text_body = wrap_text(shorten_text(str(high_row["response"]), 240), wrap_width)
        header = f"High-MQS example  ·  MQS {mqs_val:.2f}, {high_row['gen_model']}"
        # Header centred at top of the card subplot, with clear vertical gap
        # before the green-outlined box below.
        ax_text.text(
            0.5, 0.95, header,
            fontsize=14, color=GREEN, weight="bold",
            ha="center", va="top",
        )
        # Card body centred horizontally, starting well below the header so
        # the green outline does not crowd the title.
        text_kwargs = dict(
            fontsize=13, color=DARK_TEXT, ha="center", va="top",
            linespacing=1.40,
            bbox=dict(boxstyle="round,pad=0.95", linewidth=2.0,
                      edgecolor=GREEN, facecolor="white"),
        )
        if lang in ("hi", "mr") and deva_props is not None:
            text_kwargs["fontproperties"] = deva_props
        ax_text.text(0.5, 0.78, text_body, **text_kwargs)

        # Mark the example point on the scatter with a green ring
        target_x, target_y = float(high_row["u1"]), float(high_row["u2"])
        ax_scatter.scatter([target_x], [target_y], s=260, facecolors="none",
                           edgecolors=GREEN, linewidths=2.6, zorder=5)

    # Title at the top. The caption (in the LaTeX figure environment) carries
    # the colour encoding; no inline colorbar so the panel titles don't fight
    # for vertical space.
    fig.suptitle(
        "13 LLMs answer the same 380 maternal-health questions in three languages.\n"
        "Hindi and Marathi responses collapse into a much smaller region of the same shared embedding space.",
        fontsize=16, weight="bold", y=0.985,
    )

    plt.savefig(OUT_PATH, dpi=600, bbox_inches="tight", pad_inches=0.30)
    print(f"wrote {OUT_PATH.relative_to(REPO)}")


if __name__ == "__main__":
    main()
