"""Bootstrap confidence intervals and pairwise significance tests for MQS.

The body MQS table (Table~\ref{tab:mqs_overall}) reports a point estimate per
(model, dataset, language) cell with no uncertainty. This script adds:

  1. Cluster bootstrap 95% CIs per cell. We resample QUESTIONS with replacement
     (not individual responses), because the three runs of one question are not
     independent. The cell statistic is the mean over all run-level MQS values
     of the resampled questions, which matches how aggregate.py defines the cell
     mean (response-level mean, runs pooled).

  2. Pairwise model comparison within each (dataset, language). For every model
     pair we paired-bootstrap over the SHARED question set and report the mean
     MQS difference, a 95% CI, and a two-sided bootstrap p-value. We then apply
     Benjamini-Hochberg FDR across the whole family of model-pair tests in that
     (dataset, language) cell, because the table invites many simultaneous
     "X beats Y" reads.

Inputs:  runs/judge__<judge>__<gen_model>__<dataset>__<lang>.jsonl  (uses r["mqs"])
Outputs: release/results/mqs_ci__<judge>.csv
         release/results/mqs_pairwise__<judge>.csv

Deterministic: the RNG is seeded, so reruns reproduce the same intervals.

Usage:
    python -m scoring.pipeline.bootstrap_mqs --judge gpt_4o_mini --n-boot 10000
"""
from __future__ import annotations
import argparse
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scoring.pipeline import checkpoint

REPO = Path(__file__).resolve().parent.parent.parent
RUNS_DIR = REPO / "runs"
OUT = REPO / "release" / "results"
OUT.mkdir(parents=True, exist_ok=True)


def load_cell_mqs(judge_alias: str):
    """Return {(gen_model, dataset, lang): {q_idx: [mqs, ...]}}."""
    cells: dict[tuple, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for path in RUNS_DIR.glob(f"judge__{judge_alias}__*.jsonl"):
        parts = path.stem.split("__")
        if len(parts) != 5:
            continue
        gen_model, dataset, lang = parts[2], parts[3], parts[4]
        for r in checkpoint.jsonl_rows(str(path)):
            if r.get("error") or r.get("mqs") is None:
                continue
            try:
                q = int(r["q_idx"])
                m = float(r["mqs"])
            except (TypeError, ValueError, KeyError):
                continue
            cells[(gen_model, dataset, lang)][q].append(m)
    return cells


def cluster_bootstrap_ci(q_to_vals: dict[int, list[float]], n_boot: int, rng, alpha=0.05):
    """Resample questions with replacement; statistic = pooled run-level mean."""
    qs = list(q_to_vals.keys())
    vals_by_q = [np.asarray(q_to_vals[q], dtype=float) for q in qs]
    all_vals = np.concatenate(vals_by_q) if vals_by_q else np.array([])
    point = float(all_vals.mean()) if all_vals.size else float("nan")
    nq = len(qs)
    if nq < 2 or all_vals.size == 0:
        return point, point, point, 0.0, nq, int(all_vals.size)
    idx = np.arange(nq)
    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        pick = rng.choice(idx, size=nq, replace=True)
        pooled = np.concatenate([vals_by_q[i] for i in pick])
        boot[b] = pooled.mean()
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return point, lo, hi, float(boot.std(ddof=1)), nq, int(all_vals.size)


def per_question_mean(q_to_vals: dict[int, list[float]]) -> dict[int, float]:
    return {q: float(np.mean(v)) for q, v in q_to_vals.items() if v}


def paired_bootstrap_diff(a: dict[int, float], b: dict[int, float], n_boot: int, rng, alpha=0.05):
    """Paired bootstrap over the shared question set. Returns dict of stats."""
    shared = sorted(set(a) & set(b))
    if len(shared) < 2:
        return None
    da = np.array([a[q] for q in shared])
    db = np.array([b[q] for q in shared])
    diff = da - db
    obs = float(diff.mean())
    n = len(shared)
    idx = np.arange(n)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        pick = rng.choice(idx, size=n, replace=True)
        boot[i] = diff[pick].mean()
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    # two-sided bootstrap p-value: reflect the bootstrap dist about 0
    frac_le0 = float((boot <= 0).mean())
    frac_ge0 = float((boot >= 0).mean())
    p = min(1.0, 2.0 * min(frac_le0, frac_ge0))
    return {"n_shared": n, "mean_diff": obs, "ci_lo": lo, "ci_hi": hi, "p_raw": p}


def benjamini_hochberg(pvals: list[float]) -> list[float]:
    """Return BH-adjusted q-values, order preserved to match input."""
    n = len(pvals)
    if n == 0:
        return []
    order = np.argsort(pvals)
    ranked = np.asarray(pvals)[order]
    adj = ranked * n / (np.arange(n) + 1)
    # enforce monotonicity from the largest p down
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(n, dtype=float)
    out[order] = adj
    return out.tolist()


def run(judge_alias: str, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    cells = load_cell_mqs(judge_alias)
    if not cells:
        print(f"No judge files for alias '{judge_alias}'", file=sys.stderr)
        return 1

    # ---- per-cell CIs ----
    ci_rows = []
    for (m, d, l), q_to_vals in sorted(cells.items()):
        point, lo, hi, se, nq, nresp = cluster_bootstrap_ci(q_to_vals, n_boot, rng)
        ci_rows.append({
            "judge": judge_alias, "gen_model": m, "dataset": d, "lang": l,
            "mqs": round(point, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
            "ci_halfwidth": round((hi - lo) / 2, 4), "boot_se": round(se, 4),
            "n_questions": nq, "n_responses": nresp,
        })
    ci_df = pd.DataFrame(ci_rows)
    ci_path = OUT / f"mqs_ci__{judge_alias}.csv"
    ci_df.to_csv(ci_path, index=False)

    # ---- pairwise, within each (dataset, lang), with BH-FDR ----
    pair_rows = []
    by_dl: dict[tuple, list[str]] = defaultdict(list)
    for (m, d, l) in cells:
        by_dl[(d, l)].append(m)
    for (d, l), models in sorted(by_dl.items()):
        models = sorted(models)
        family = []
        for ma, mb in combinations(models, 2):
            a = per_question_mean(cells[(ma, d, l)])
            b = per_question_mean(cells[(mb, d, l)])
            res = paired_bootstrap_diff(a, b, n_boot, rng)
            if res is None:
                continue
            family.append({"dataset": d, "lang": l, "model_a": ma, "model_b": mb, **res})
        qvals = benjamini_hochberg([row["p_raw"] for row in family])
        for row, q in zip(family, qvals):
            row["p_bh_fdr"] = round(q, 4)
            row["significant_fdr_05"] = bool(q < 0.05)
            row["mean_diff"] = round(row["mean_diff"], 4)
            row["ci_lo"] = round(row["ci_lo"], 4)
            row["ci_hi"] = round(row["ci_hi"], 4)
            row["p_raw"] = round(row["p_raw"], 4)
        pair_rows.extend(family)
    pair_df = pd.DataFrame(pair_rows)
    pair_path = OUT / f"mqs_pairwise__{judge_alias}.csv"
    pair_df.to_csv(pair_path, index=False)

    # ---- console summary ----
    print(f"[bootstrap_mqs] judge={judge_alias} n_boot={n_boot} seed={seed}")
    print(f"  cells: {len(ci_df)}  -> {ci_path}")
    if len(ci_df):
        print(f"  median CI half-width: {ci_df['ci_halfwidth'].median():.4f} MQS")
    print(f"  model pairs: {len(pair_df)}  -> {pair_path}")
    if len(pair_df):
        sig = int(pair_df["significant_fdr_05"].sum())
        print(f"  pairs significant after BH-FDR (q<0.05): {sig}/{len(pair_df)} "
              f"({100*sig/len(pair_df):.0f}%)")
        small = pair_df[pair_df["mean_diff"].abs() < 0.05]
        if len(small):
            sig_small = int(small["significant_fdr_05"].sum())
            print(f"  of pairs with |diff|<0.05 MQS: {sig_small}/{len(small)} survive FDR")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="gpt_4o_mini",
                    help="judge alias, matches judge__<alias>__... filenames")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260601)
    args = ap.parse_args()
    sys.exit(run(args.judge, args.n_boot, args.seed))


if __name__ == "__main__":
    main()
