"""Model x Model pairwise response-similarity heatmap, Artificial-Hivemind Fig 6 style.

For each question (fixed dataset, lang, q_idx) we have one response per model. We compute
pairwise cosine similarity between every pair of models on that question, then average across
all questions to get a 13x13 model-vs-model matrix. The matrix is rendered as a lower-triangular
heatmap with the on-diagonal cells (a model's similarity to itself) omitted.

The point of the figure: when two models give substantially-similar responses to the same
question on average, that is evidence for response-distribution collapse across models. Low
off-diagonal similarity, by contrast, indicates that models give meaningfully different
answers and that the benchmark separates them.

Defaults: dataset = non_expert, lang = en (the deployment-facing arm).

Output: overleaf/paper/pictures/fig_model_similarity_heatmap.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO = Path(__file__).resolve().parents[2]
EMBED_PATH = REPO / "data" / "embeddings" / "embeddings_run1.parquet"
OUT_PATH = REPO / "overleaf" / "paper" / "pictures" / "fig_model_similarity_heatmap.pdf"

MODEL_DISPLAY = {
    "claude_opus_4_7": "Claude Opus 4.7",
    "claude_haiku_4_5": "Claude Haiku 4.5",
    "gpt_5_mini": "GPT-5 Mini",
    "gpt_4o_mini": "GPT-4o Mini",
    "gemini_3_pro": "Gemini 3 Pro",
    "gemini_3_flash": "Gemini 3 Flash",
    "gemini_3_1_flash_lite": "Gemini 3.1 Flash-Lite",
    "llama_3_3_70b": "Llama 3.3 70B",
    "llama_4_maverick": "Llama 4 Maverick",
    "cohere_command_a": "Command A",
    "aya_expanse": "Aya Expanse",
    "gemma_3_27b": "Gemma 3 27B",
    "medgemma_27b": "MedGemma 27B",
    "medgemma_4b": "MedGemma 4B",
}


def load_embeddings(dataset: str, lang: str) -> pd.DataFrame:
    df = pd.read_parquet(EMBED_PATH)
    sub = df[(df["dataset"] == dataset) & (df["lang"] == lang)].copy()
    return sub


def cosine_sim_matrix(vecs: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity for a stack of (n, d) embeddings."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = vecs / norms
    return unit @ unit.T


def model_pair_similarity(df: pd.DataFrame, models: list[str]) -> tuple[np.ndarray, dict]:
    """Average pairwise cosine similarity between every pair of models, across questions.

    For each q_idx, we collect the per-model embedding (one per model that produced a valid
    response for this question, run=1). Then we compute the similarity matrix over those models
    for that question and accumulate. Final per-pair value is the mean over the questions where
    both models in the pair were present.
    """
    n = len(models)
    sum_sim = np.zeros((n, n))
    count = np.zeros((n, n))
    by_q = df.groupby("q_idx")
    for _, grp in by_q:
        present = [m for m in models if m in set(grp["gen_model"])]
        if len(present) < 2:
            continue
        emb_by_model = {row.gen_model: np.asarray(row.embedding) for row in grp.itertuples()}
        present_vecs = np.stack([emb_by_model[m] for m in present])
        sim = cosine_sim_matrix(present_vecs)
        for i, mi in enumerate(present):
            ii = models.index(mi)
            for j, mj in enumerate(present):
                jj = models.index(mj)
                sum_sim[ii, jj] += sim[i, j]
                count[ii, jj] += 1
    with np.errstate(invalid="ignore"):
        avg = sum_sim / count
    avg[np.isnan(avg)] = 0.0
    return avg, {"n_questions": int(by_q.ngroups), "n_models_present": int((count.diagonal() > 0).sum())}


def render(dataset: str, lang: str, out_path: Path):
    df = load_embeddings(dataset, lang)
    if df.empty:
        print(f"no embeddings for {dataset}/{lang}")
        return
    counts = df["gen_model"].value_counts()
    models_present = counts[counts > 0].index.tolist()
    # Order: roughly group by training family (proprietary first, then open-weight, then medical)
    family_order = [
        "claude_opus_4_7", "claude_haiku_4_5",
        "gpt_5_mini", "gpt_4o_mini",
        "gemini_3_pro", "gemini_3_flash", "gemini_3_1_flash_lite",
        "cohere_command_a",
        "llama_3_3_70b", "llama_4_maverick", "aya_expanse",
        "gemma_3_27b", "medgemma_27b", "medgemma_4b",
    ]
    models = [m for m in family_order if m in models_present]
    print(f"{dataset}/{lang}: {len(models)} models with embeddings")

    avg, meta = model_pair_similarity(df, models)
    print(f"  averaged over {meta['n_questions']} questions")

    labels = [MODEL_DISPLAY.get(m, m) for m in models]

    # Lower-triangular mask: hide diagonal and upper triangle.
    mask = np.triu(np.ones_like(avg, dtype=bool), k=0)

    fig, ax = plt.subplots(figsize=(10.5, 9.0))
    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    vmin = float(np.percentile(avg[~mask], 5))
    vmax = float(np.percentile(avg[~mask], 95))
    sns.heatmap(
        avg,
        mask=mask,
        annot=True, fmt=".2f",
        cmap=cmap, vmin=vmin, vmax=vmax,
        xticklabels=labels, yticklabels=labels,
        cbar_kws={"label": "Mean pairwise cosine similarity (response embeddings)"},
        annot_kws={"size": 9},
        ax=ax,
        square=True,
        linewidths=0.4, linecolor="white",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    ax.set_title(
        f"Model-vs-model response similarity ({dataset.replace('_', '-')}, {lang.upper()}, run 1)\n"
        f"averaged over {meta['n_questions']} questions",
        fontsize=12, weight="bold", pad=12,
    )
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="non_expert", choices=["expert", "non_expert"])
    ap.add_argument("--lang", default="en", choices=["en", "hi", "mr"])
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()
    render(args.dataset, args.lang, Path(args.out))


if __name__ == "__main__":
    main()
