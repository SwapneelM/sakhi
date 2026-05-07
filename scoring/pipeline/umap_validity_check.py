"""
Sanity-check the UMAP-spread signal in fig_response_embedding_clusters.

The header figure's narrative is: "English responses are spread out;
Hindi and Marathi responses are tightly clustered, suggesting the model
panel is producing more generic content in non-English languages."

Two ways that signal could be a bug instead of a finding:
  (1) UMAP-method bias: spread differences in the 2-D layout are an
      artefact of nonlinear dimensionality reduction. Same-language
      responses might genuinely be no closer in the original 1536-dim
      space than cross-language responses are.
  (2) Embedding-model bias: text-embedding-3-small is trained mostly on
      English, so it might compress Hindi and Marathi text into a denser
      region of representation regardless of actual response diversity.

This script tests both:
  - Stage A: compute language-wise spread metrics IN THE RAW 1536-DIM SPACE
    (no UMAP). If the spread collapse persists, UMAP is not the cause.
  - Stage B: re-embed a balanced sample with a multilingual sentence-
    transformer (paraphrase-multilingual-mpnet-base-v2, trained on 50+
    languages with explicit cross-lingual alignment). If the spread
    collapse persists, embedding-model bias is not the cause.

Outputs:
  - data/umap_validity_metrics.csv (per-language metrics from both stages)
  - scratchpad/umap-validity-v1.md (writeup)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EMBED_PATH = ROOT / "data" / "embeddings" / "embeddings_run1.parquet"
OUT_METRICS = ROOT / "data" / "umap_validity_metrics.csv"
SAMPLE_PARQUET = ROOT / "data" / "embeddings" / "embeddings_mpnet_sample.parquet"


def per_language_spread(df: pd.DataFrame, embedding_col: str) -> pd.DataFrame:
    """Compute spread metrics per language. NO UMAP — operates in raw embedding space."""
    rows = []
    for lang, sub in df.groupby("lang"):
        X = np.stack(sub[embedding_col].values)  # (n, d)
        # L2-normalise so cosine and Euclidean agree
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        n, d = X.shape
        centroid = X.mean(axis=0)
        # mean distance to centroid (lower = tighter cluster)
        dist_to_centroid = np.linalg.norm(X - centroid, axis=1)
        # mean pairwise cosine distance via dot products on a sample if too big
        if n > 4000:
            idx = np.random.RandomState(0).choice(n, 4000, replace=False)
            Xs = X[idx]
        else:
            Xs = X
        # cosine sim matrix; convert to distance = 1 - sim; exclude diagonal
        sim = Xs @ Xs.T
        np.fill_diagonal(sim, np.nan)
        mean_pair_cos_sim = np.nanmean(sim)
        rows.append({
            "lang": lang,
            "n": n,
            "dim": d,
            "mean_dist_to_centroid": float(dist_to_centroid.mean()),
            "median_dist_to_centroid": float(np.median(dist_to_centroid)),
            "p90_dist_to_centroid": float(np.percentile(dist_to_centroid, 90)),
            "var_per_dim": float(X.var(axis=0).mean()),
            "mean_pairwise_cos_sim": float(mean_pair_cos_sim),
            "mean_pairwise_cos_dist": float(1 - mean_pair_cos_sim),
        })
    return pd.DataFrame(rows).sort_values("lang").reset_index(drop=True)


def cross_language_distance(df: pd.DataFrame, embedding_col: str) -> dict:
    """For each language pair, compute mean pairwise cosine distance between
    paired (same q_idx, same gen_model, different lang) embeddings.

    If text-embedding-3-small treats Hindi and Marathi as 'foreign-language
    blob' regardless of content, paired Hindi and Marathi responses should
    be MORE similar than paired English-Hindi responses, even when the
    English-Hindi pair is talking about the same question.
    """
    pivot = df.pivot_table(
        index=["gen_model", "q_idx"], columns="lang", values=embedding_col, aggfunc="first"
    )
    pivot = pivot.dropna(subset=["en", "hi", "mr"])
    out = {}
    for (a, b) in [("en", "hi"), ("en", "mr"), ("hi", "mr")]:
        Xa = np.stack(pivot[a].values)
        Xb = np.stack(pivot[b].values)
        Xa = Xa / np.linalg.norm(Xa, axis=1, keepdims=True)
        Xb = Xb / np.linalg.norm(Xb, axis=1, keepdims=True)
        sims = (Xa * Xb).sum(axis=1)
        out[f"{a}-{b}"] = {
            "n_pairs": int(len(sims)),
            "mean_cos_sim": float(sims.mean()),
            "mean_cos_dist": float(1 - sims.mean()),
        }
    return out


def stage_a_raw_openai(df: pd.DataFrame) -> pd.DataFrame:
    print("=== Stage A: per-language spread in raw 1536-dim OpenAI space ===")
    metrics = per_language_spread(df, "embedding")
    metrics["embedding_model"] = "openai/text-embedding-3-small"
    print(metrics.to_string(index=False))
    print("\n--- cross-language paired distances (raw OpenAI) ---")
    print(json.dumps(cross_language_distance(df, "embedding"), indent=2))
    return metrics


def stage_b_mpnet(df: pd.DataFrame, sample_per_lang: int = 1000) -> pd.DataFrame | None:
    """Re-embed a balanced sample with paraphrase-multilingual-mpnet-base-v2."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("sentence_transformers not installed; skipping Stage B")
        print("install with: pip install sentence-transformers")
        return None

    if SAMPLE_PARQUET.exists():
        print(f"=== Stage B: loading cached mpnet embeddings from {SAMPLE_PARQUET.name} ===")
        sample = pd.read_parquet(SAMPLE_PARQUET)
    else:
        print(f"=== Stage B: embedding {sample_per_lang} per language with mpnet ===")
        rng = np.random.RandomState(42)
        parts = []
        for lang, sub in df.groupby("lang"):
            n = min(sample_per_lang, len(sub))
            idx = rng.choice(len(sub), n, replace=False)
            parts.append(sub.iloc[idx])
        sample = pd.concat(parts, ignore_index=True).copy()

        model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        texts = sample["response"].astype(str).tolist()
        print(f"  encoding {len(texts)} texts ...")
        t0 = time.time()
        new_embs = model.encode(
            texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True
        )
        print(f"  done in {time.time()-t0:.1f}s (dim={new_embs.shape[1]})")
        sample["embedding_mpnet"] = list(new_embs)
        sample.to_parquet(SAMPLE_PARQUET)

    metrics = per_language_spread(sample, "embedding_mpnet")
    metrics["embedding_model"] = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    print(metrics.to_string(index=False))
    print("\n--- cross-language paired distances (mpnet, sampled) ---")
    print(json.dumps(cross_language_distance(sample, "embedding_mpnet"), indent=2))
    return metrics


def main():
    print(f"loading {EMBED_PATH.name} ...")
    df = pd.read_parquet(EMBED_PATH)
    print(f"  rows: {len(df)}, languages: {df['lang'].value_counts().to_dict()}")

    metrics_a = stage_a_raw_openai(df)
    metrics_b = stage_b_mpnet(df)

    parts = [metrics_a]
    if metrics_b is not None:
        parts.append(metrics_b)
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(OUT_METRICS, index=False)
    print(f"\nwrote metrics -> {OUT_METRICS}")


if __name__ == "__main__":
    main()
