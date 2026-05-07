"""
Embed the Sakhi questions (NOT answers) in English, Hindi, and Marathi using
the SAME embedding model we used for answers (text-embedding-3-small via
OpenRouter), then compute the same per-language spread metrics.

Why this matters: our header figure says answer embeddings collapse in HI/MR
relative to EN. This is only a story about model output if the QUESTIONS,
embedded the same way, do NOT collapse. If the questions also collapse in
HI/MR, then what we are seeing is partly the embedding model treating
Devanagari text as a tighter region of vector space, and the finding is
weaker. If the questions stay spread out, the collapse is real.

Inputs:
    data2/sakhi_expert_raw_150.csv         (149 expert questions × 3 langs)
    data2/sakhi_non_expert_raw_230.csv     (231 non-expert questions × 3 langs)

Outputs:
    data/embeddings/embeddings_questions.parquet
    data/question_spread_metrics.csv (per-language spread, both arms combined)
    scratchpad/question-vs-answer-spread-v1.md
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[2]
load_dotenv(Path.home() / ".env.openrouter")
OR_KEY = os.environ.get("OPENROUTER_API_KEY", "")
if not OR_KEY:
    sys.exit("OPENROUTER_API_KEY missing")

EMBED_OUT = REPO / "data" / "embeddings" / "embeddings_questions.parquet"
EMBED_OUT.parent.mkdir(parents=True, exist_ok=True)
SPREAD_OUT = REPO / "data" / "question_spread_metrics.csv"


def load_all_questions() -> pd.DataFrame:
    rows = []
    expert = pd.read_csv(REPO / "data2" / "sakhi_expert_raw_150.csv")
    for i, r in expert.iterrows():
        for lang, col in [("en", "question"), ("hi", "question_hi"), ("mr", "question_mr")]:
            q = r.get(col)
            if pd.isna(q) or not str(q).strip():
                continue
            rows.append({
                "dataset": "expert",
                "lang": lang,
                "q_idx": int(i),
                "q_id": str(r.get("q_id", "")),
                "question": str(q).strip(),
            })
    nonexpert = pd.read_csv(REPO / "data2" / "sakhi_non_expert_raw_230.csv")
    for i, r in nonexpert.iterrows():
        for lang, col in [("en", "question"), ("hi", "questions_hindi"), ("mr", "questions_marathi")]:
            q = r.get(col)
            if pd.isna(q) or not str(q).strip():
                continue
            rows.append({
                "dataset": "non_expert",
                "lang": lang,
                "q_idx": int(i),
                "q_id": "",
                "question": str(q).strip(),
            })
    return pd.DataFrame(rows)


def batch_embed(texts: list[str], batch_size: int = 100) -> np.ndarray:
    """Use OpenAI text-embedding-3-small via OpenRouter. Same model as answers."""
    url = "https://openrouter.ai/api/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {OR_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/SwapneelM/MedicaLLM-Eval",
        "X-Title": "Sakhi question embeddings",
    }
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        payload = {"model": "openai/text-embedding-3-small", "input": chunk}
        for attempt in range(3):
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=120)
                r.raise_for_status()
                data = r.json()
                vecs = [d["embedding"] for d in data["data"]]
                out.extend(vecs)
                print(f"  embedded {i + len(chunk)}/{len(texts)}", flush=True)
                break
            except Exception as e:
                print(f"  attempt {attempt} failed: {e}", flush=True)
                time.sleep(2 ** attempt)
        else:
            sys.exit("3 attempts failed")
    return np.array(out, dtype=np.float32)


def per_language_spread(emb_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lang, sub in emb_df.groupby("lang"):
        X = np.stack(sub["embedding"].values)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        n, d = X.shape
        centroid = X.mean(axis=0)
        dist = np.linalg.norm(X - centroid, axis=1)
        if n > 4000:
            idx = np.random.RandomState(0).choice(n, 4000, replace=False)
            Xs = X[idx]
        else:
            Xs = X
        sim = Xs @ Xs.T
        np.fill_diagonal(sim, np.nan)
        rows.append({
            "lang": lang,
            "n": n,
            "mean_dist_to_centroid": float(dist.mean()),
            "median_dist_to_centroid": float(np.median(dist)),
            "p90_dist_to_centroid": float(np.percentile(dist, 90)),
            "mean_pairwise_cos_dist": float(1 - np.nanmean(sim)),
        })
    return pd.DataFrame(rows).sort_values("lang").reset_index(drop=True)


def main():
    if EMBED_OUT.exists():
        print(f"loading cached {EMBED_OUT.name}")
        df = pd.read_parquet(EMBED_OUT)
    else:
        df = load_all_questions()
        # Dedup on question text within each lang to avoid double-counting non-expert duplicates
        df = df.drop_duplicates(["lang", "question"]).reset_index(drop=True)
        print(f"{len(df)} unique questions to embed across EN/HI/MR")
        embeddings = batch_embed(df["question"].tolist())
        df["embedding"] = list(embeddings)
        df.to_parquet(EMBED_OUT)
        print(f"wrote {EMBED_OUT}")

    metrics = per_language_spread(df)
    metrics["source"] = "questions"
    metrics["embedding_model"] = "openai/text-embedding-3-small"
    print("\n=== Question-embedding spread per language (no UMAP) ===")
    print(metrics.to_string(index=False))

    # Compare against the answer-spread metrics we already have
    ans_metrics_path = REPO / "data" / "umap_validity_metrics.csv"
    if ans_metrics_path.exists():
        a = pd.read_csv(ans_metrics_path)
        a = a[a["embedding_model"] == "openai/text-embedding-3-small"].copy()
        a["source"] = "answers"
        a = a[["lang", "n", "mean_dist_to_centroid", "p90_dist_to_centroid", "mean_pairwise_cos_dist", "embedding_model", "source"]]
        combined = pd.concat([a, metrics[a.columns]], ignore_index=True)
        combined.to_csv(SPREAD_OUT, index=False)
        print("\n=== Side-by-side: answer spread vs question spread (raw OpenAI 1536-dim) ===")
        pivot = combined.pivot_table(index=["source", "lang"],
                                     values=["mean_dist_to_centroid", "mean_pairwise_cos_dist"]).round(4)
        print(pivot)
        print(f"\nwrote {SPREAD_OUT}")


if __name__ == "__main__":
    main()
