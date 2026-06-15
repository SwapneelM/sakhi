"""Controls for the cross-lingual embedding-spread claim.

The header figure shows that Hindi and Marathi responses occupy a smaller region
of embedding space than English. On its own that is a statement about diversity,
not quality. This script runs the controls a referee will ask for before anyone
reads the smaller region as "worse":

  1. Per-language and per-(model, language) spread in the raw 1536-dim space
     (mean pairwise cosine distance, mean distance to the language centroid),
     matching umap_validity_check.per_language_spread.

  2. Response length per language (a known confound: shorter text embeds tighter).

  3. Length-stratified spread: bin every response into global word-count quartiles
     and recompute per-language spread inside each bin. If English > Hindi > Marathi
     survives within matched length bins, length is not what drives the gap.

  4. Lexical diversity that confounds differently from embeddings: distinct-1,
     distinct-2, and a sampled self-BLEU per language.

  5. The actual diversity-to-quality test. Two reads:
       (a) per response, Spearman between distance-to-centroid (typicality) and MQS;
       (b) per (model, language) cell, Spearman between spread and mean MQS.
     If "less diverse = worse" were true, tighter/typical responses would score
     lower (positive correlation). If the correlation is ~0, lower diversity is
     not evidence of lower quality.

Inputs:  data/embeddings/embeddings_run1.parquet  (gen_model, dataset, lang, run,
                                                    q_idx, response, embedding)
         runs/judge__<judge>__*.jsonl             (per-response mqs, run 1)
Outputs: release/results/diversity_*.csv

Deterministic (seeded). Usage:
    python -m scoring.pipeline.diversity_controls --judge gpt_4o_mini
"""
from __future__ import annotations
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scoring.pipeline import checkpoint

REPO = Path(__file__).resolve().parent.parent.parent
RUNS_DIR = REPO / "runs"
EMB_PATH = REPO / "data" / "embeddings" / "embeddings_run1.parquet"
OUT = REPO / "release" / "results"
OUT.mkdir(parents=True, exist_ok=True)
SAMPLE_PAIRWISE = 4000   # cap for the pairwise-cosine computation, matches umap check
SELFBLEU_SAMPLE = 200


def _norm(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def spread_metrics(X: np.ndarray, rng) -> dict:
    """mean distance to centroid and mean pairwise cosine distance, in raw space."""
    if len(X) < 2:
        return {"n": len(X), "mean_dist_to_centroid": float("nan"),
                "mean_pairwise_cos_dist": float("nan")}
    Xn = _norm(X.astype(float))
    centroid = Xn.mean(axis=0)
    dist_to_centroid = np.linalg.norm(Xn - centroid, axis=1)
    if len(Xn) > SAMPLE_PAIRWISE:
        idx = rng.choice(len(Xn), SAMPLE_PAIRWISE, replace=False)
        Xs = Xn[idx]
    else:
        Xs = Xn
    sim = Xs @ Xs.T
    np.fill_diagonal(sim, np.nan)
    mean_pair_sim = float(np.nanmean(sim))
    return {"n": int(len(X)),
            "mean_dist_to_centroid": float(dist_to_centroid.mean()),
            "mean_pairwise_cos_dist": float(1 - mean_pair_sim)}


def spearman(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return float("nan")
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom else float("nan")


def word_tokens(text: str) -> list[str]:
    # works for Devanagari too: Hindi/Marathi separate words with spaces
    return re.findall(r"\S+", str(text))


def distinct_n(corpus_tokens: list[list[str]], n: int) -> float:
    total, seen = 0, set()
    for toks in corpus_tokens:
        grams = [tuple(toks[i:i+n]) for i in range(len(toks) - n + 1)]
        total += len(grams)
        seen.update(grams)
    return len(seen) / total if total else float("nan")


def self_bleu(texts: list[str], rng) -> float:
    try:
        import sacrebleu
    except Exception:
        return float("nan")
    texts = [t for t in texts if str(t).strip()]
    if len(texts) < 5:
        return float("nan")
    if len(texts) > SELFBLEU_SAMPLE:
        idx = rng.choice(len(texts), SELFBLEU_SAMPLE, replace=False)
        texts = [texts[i] for i in idx]
    scores = []
    for i, hyp in enumerate(texts):
        refs = [texts[j] for j in range(len(texts)) if j != i]
        # one ref per call keeps memory bounded; sample 20 refs for speed
        ref_sample = [refs[k] for k in rng.choice(len(refs), min(20, len(refs)), replace=False)]
        scores.append(sacrebleu.sentence_bleu(hyp, ref_sample).score)
    return float(np.mean(scores)) / 100.0


def load_mqs_run1(judge_alias: str) -> dict[tuple, float]:
    """{(gen_model, dataset, lang, q_idx): mqs} from run 1 to align with embeddings."""
    out: dict[tuple, float] = {}
    for path in RUNS_DIR.glob(f"judge__{judge_alias}__*.jsonl"):
        parts = path.stem.split("__")
        if len(parts) != 5:
            continue
        m, d, l = parts[2], parts[3], parts[4]
        for r in checkpoint.jsonl_rows(str(path)):
            if r.get("error") or r.get("mqs") is None or r.get("run") != 1:
                continue
            try:
                out[(m, d, l, int(r["q_idx"]))] = float(r["mqs"])
            except (TypeError, ValueError, KeyError):
                continue
    return out


def run(judge_alias: str, seed: int):
    rng = np.random.default_rng(seed)
    if not EMB_PATH.exists():
        print(f"FATAL: embeddings not found at {EMB_PATH}", file=sys.stderr)
        return 1
    df = pd.read_parquet(EMB_PATH)
    df = df[df["run"] == 1].copy() if "run" in df.columns else df
    df["n_words"] = df["response"].map(lambda t: len(word_tokens(t)))
    df["n_chars"] = df["response"].map(lambda t: len(str(t)))

    # ---- 1+2: per-language spread + length ----
    lang_rows = []
    for lang, sub in df.groupby("lang"):
        X = np.stack(sub["embedding"].values)
        sm = spread_metrics(X, rng)
        toks = [word_tokens(t) for t in sub["response"]]
        lang_rows.append({
            "lang": lang, **sm,
            "mean_words": round(sub["n_words"].mean(), 1),
            "median_words": int(sub["n_words"].median()),
            "mean_chars": round(sub["n_chars"].mean(), 1),
            "distinct_1": round(distinct_n(toks, 1), 4),
            "distinct_2": round(distinct_n(toks, 2), 4),
            "self_bleu": round(self_bleu(list(sub["response"]), rng), 4),
        })
    pd.DataFrame(lang_rows).to_csv(OUT / "diversity_per_language.csv", index=False)

    # ---- per (model, language) spread ----
    ml_rows = []
    for (m, lang), sub in df.groupby(["gen_model", "lang"]):
        X = np.stack(sub["embedding"].values)
        sm = spread_metrics(X, rng)
        ml_rows.append({"gen_model": m, "lang": lang, **sm,
                        "mean_words": round(sub["n_words"].mean(), 1)})
    ml_df = pd.DataFrame(ml_rows)
    ml_df.to_csv(OUT / "diversity_per_model_language.csv", index=False)

    # ---- 3: length-stratified spread (global word-count quartiles) ----
    qs = df["n_words"].quantile([0.25, 0.5, 0.75]).tolist()
    def wbin(w):
        return 0 if w <= qs[0] else 1 if w <= qs[1] else 2 if w <= qs[2] else 3
    df["wbin"] = df["n_words"].map(wbin)
    strat_rows = []
    for (b, lang), sub in df.groupby(["wbin", "lang"]):
        X = np.stack(sub["embedding"].values)
        sm = spread_metrics(X, rng)
        strat_rows.append({"word_quartile": int(b), "lang": lang, **sm,
                           "mean_words": round(sub["n_words"].mean(), 1)})
    pd.DataFrame(strat_rows).to_csv(OUT / "diversity_length_stratified.csv", index=False)

    # ---- 5: diversity -> quality ----
    mqs = load_mqs_run1(judge_alias)
    df["mqs"] = [mqs.get((r.gen_model, r.dataset, r.lang, int(r.q_idx)))
                 for r in df.itertuples(index=False)]
    corr_rows = []
    # (a) per-response typicality (dist to lang centroid) vs MQS, per language
    for lang, sub in df.groupby("lang"):
        s = sub.dropna(subset=["mqs"])
        if len(s) < 10:
            continue
        Xn = _norm(np.stack(s["embedding"].values).astype(float))
        centroid = Xn.mean(axis=0)
        dist = np.linalg.norm(Xn - centroid, axis=1)
        rho = spearman(dist, s["mqs"].to_numpy())
        corr_rows.append({"scope": "per_response_typicality_vs_mqs", "lang": lang,
                          "n": int(len(s)), "spearman_rho": round(rho, 4),
                          "note": "positive => responses far from centroid (more distinctive) score higher"})
    # (b) per (model,lang) cell spread vs mean MQS, per language
    cell = (df.dropna(subset=["mqs"])
              .groupby(["gen_model", "lang"])
              .agg(mean_mqs=("mqs", "mean")).reset_index())
    cell = cell.merge(ml_df[["gen_model", "lang", "mean_pairwise_cos_dist"]],
                      on=["gen_model", "lang"], how="left")
    for lang, sub in cell.groupby("lang"):
        if len(sub) < 4:
            continue
        rho = spearman(sub["mean_pairwise_cos_dist"], sub["mean_mqs"])
        corr_rows.append({"scope": "per_cell_spread_vs_mean_mqs", "lang": lang,
                          "n": int(len(sub)), "spearman_rho": round(rho, 4),
                          "note": "positive => higher-spread models score higher"})
    pd.DataFrame(corr_rows).to_csv(OUT / "diversity_quality_correlation.csv", index=False)

    # ---- console summary ----
    print(f"[diversity_controls] judge={judge_alias} seed={seed}  (run-1 responses: {len(df)})")
    ld = pd.DataFrame(lang_rows).set_index("lang")
    print("\nPer-language spread + length + lexical diversity:")
    print(ld[["mean_pairwise_cos_dist", "mean_words", "distinct_2", "self_bleu"]].round(4).to_string())
    print("\nLength-stratified spread (mean pairwise cos dist by word quartile):")
    piv = (pd.DataFrame(strat_rows)
           .pivot(index="word_quartile", columns="lang", values="mean_pairwise_cos_dist"))
    print(piv.round(4).to_string())
    print("\nDiversity -> quality (Spearman):")
    print(pd.DataFrame(corr_rows)[["scope", "lang", "n", "spearman_rho"]].to_string(index=False))
    print(f"\nWrote 4 CSVs to {OUT}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="gpt_4o_mini")
    ap.add_argument("--seed", type=int, default=20260601)
    args = ap.parse_args()
    sys.exit(run(args.judge, args.seed))


if __name__ == "__main__":
    main()
