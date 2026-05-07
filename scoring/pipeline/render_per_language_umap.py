"""
Render the per-language UMAP comparison figure for the Sakhi paper appendix.

This is the companion figure to fig_response_embedding_clusters.pdf.
The headline figure runs ONE shared UMAP across all three languages, so
its claim "Hindi responses cover a smaller region of the same space"
relies on a single shared coordinate frame. A reviewer could reasonably
ask: what if we ran UMAP independently for each language? Would the
internal structure look different?

This script answers that. We fit three separate UMAPs (one per language)
and plot each on its own axes. The coordinate frames are NOT comparable
across panels (each UMAP picks its own axes), so the "smaller region"
claim does not apply here; what this figure shows is whether internal
structure within each language reveals tighter or looser clustering.

Output:
    overleaf/paper/pictures/fig_response_embedding_per_lang.pdf
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import umap

REPO = Path(__file__).resolve().parents[2]
EMBED_PATH = REPO / "data" / "embeddings" / "embeddings_run1.parquet"
RUNS_DIR = REPO / "runs"
OUT_PATH = REPO / "overleaf" / "paper" / "pictures" / "fig_response_embedding_per_lang.pdf"


def load_mqs_lookup() -> dict:
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


def main():
    print("loading embeddings + MQS ...")
    df = pd.read_parquet(EMBED_PATH).copy()
    mqs_lookup = load_mqs_lookup()
    df["mqs"] = df.apply(
        lambda r: mqs_lookup.get((r["gen_model"], r["dataset"], r["lang"], int(r["run"]), int(r["q_idx"]))),
        axis=1,
    )
    df = df.dropna(subset=["mqs"]).reset_index(drop=True)

    plt.rcParams.update({
        "font.size": 12, "font.family": "DejaVu Sans",
        "axes.titlesize": 13, "axes.labelsize": 11,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.dpi": 600, "figure.dpi": 200,
    })

    fig = plt.figure(figsize=(15.0, 5.5))
    gs = GridSpec(1, 3, wspace=0.22, left=0.05, right=0.97, top=0.82, bottom=0.20)

    cmap = plt.get_cmap("RdYlGn")
    # Compute global vmin/vmax from ALL responses so colour-by-MQS is comparable
    vmin = float(df["mqs"].quantile(0.05))
    vmax = float(df["mqs"].quantile(0.95))
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    lang_titles = {
        "en": "English (independent UMAP)",
        "hi": "Hindi (independent UMAP)",
        "mr": "Marathi (independent UMAP)",
    }

    for col, lang in enumerate(["en", "hi", "mr"]):
        sub = df[df["lang"] == lang].copy()
        X = np.stack(sub["embedding"].values)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        print(f"  fitting UMAP on {lang} (n={len(sub)}) ...")
        reducer = umap.UMAP(n_neighbors=20, min_dist=0.05, n_components=2,
                            metric="cosine", random_state=42)
        coords = reducer.fit_transform(X)
        sub["u1"], sub["u2"] = coords[:, 0], coords[:, 1]

        ax = fig.add_subplot(gs[0, col])
        order = sub["mqs"].rank(method="first").values
        sub = sub.iloc[np.argsort(order)]
        ax.scatter(sub["u1"], sub["u2"], c=sub["mqs"], cmap=cmap, norm=norm,
                   s=80, alpha=0.75, edgecolor="none")
        ax.set_title(f"{lang_titles[lang]} (n={len(sub)})",
                     fontsize=13, weight="bold", loc="left", pad=8)
        ax.set_xlabel("UMAP-1", fontsize=11)
        if col == 0:
            ax.set_ylabel("UMAP-2", fontsize=11)
        ax.tick_params(axis="both", which="both", labelsize=9)

    fig.suptitle(
        "Per-language UMAP (each panel fit independently; coordinate frames NOT comparable across panels)",
        fontsize=12, y=0.97,
    )

    cbar_ax = fig.add_axes([0.32, 0.04, 0.36, 0.018])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Medical Quality Score (red = poor, green = good)",
                   fontsize=9.5, color="#1a1a1a", labelpad=2)
    cbar.ax.tick_params(labelsize=8)

    plt.savefig(OUT_PATH, dpi=600, bbox_inches="tight", pad_inches=0.25)
    print(f"wrote {OUT_PATH.relative_to(REPO)}")


if __name__ == "__main__":
    main()
