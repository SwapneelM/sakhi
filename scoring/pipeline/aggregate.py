"""Aggregate per-response JSONLs into the summary tables the paper reports.

Outputs (in release/results/):
  - mqs_per_model.csv           model × dataset × lang → mean MQS, std, n
  - mqs_per_theme.csv           model × dataset × lang × theme → mean MQS
  - axis_per_model.csv          model × dataset × lang × axis → mean pass rate
  - metrics_per_model.csv       model × dataset × lang → mean BLEU, chrF++, METEOR,
                                 ROUGE-L, BERTScore F1, embedding similarity
  - valid_response_counts.csv   generation reliability per (model × dataset × lang × run)
"""
from __future__ import annotations
import json, sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scoring.pipeline import checkpoint
from scoring.scoring_rubric import AXIS_MAP

REPO = Path(__file__).resolve().parent.parent.parent
RUNS_DIR = REPO / "runs"
OUT = REPO / "release" / "results"
OUT.mkdir(parents=True, exist_ok=True)

AXIS_WEIGHTS = {"Accuracy": 0.30, "Completeness": 0.25, "Context Awareness": 0.20,
                "Communication": 0.15, "Terminology Accessibility": 0.10}


def aggregate_judge(judge_alias: str = "gpt_4o_mini"):
    mqs_rows = []
    theme_mqs_rows = []
    axis_rows = []
    by_key: dict[tuple, list[float]] = defaultdict(list)
    theme_key: dict[tuple, list[float]] = defaultdict(list)
    axis_pass: dict[tuple, list[int]] = defaultdict(list)

    for path in RUNS_DIR.glob(f"judge__{judge_alias}__*.jsonl"):
        parts = path.stem.split("__")
        # judge__<judge_alias>__<gen_model>__<dataset>__<lang>
        if len(parts) != 5:
            continue
        gen_model = parts[2]; dataset = parts[3]; lang = parts[4]
        for r in checkpoint.jsonl_rows(str(path)):
            if r.get("error") or r.get("mqs") is None:
                continue
            mqs = float(r["mqs"])
            by_key[(gen_model, dataset, lang)].append(mqs)
            theme_key[(gen_model, dataset, lang, r.get("theme", ""))].append(mqs)
            # axis pass rates
            scores = r.get("rubric_scores") or {}
            theme = r.get("theme", "")
            ax = AXIS_MAP.get(theme)
            if ax:
                try:
                    cat_map = json.loads(ax)
                except Exception:
                    cat_map = {}
                # Normalize keys (case-insensitive)
                norm = {k.strip().lower(): v for k, v in scores.items()}
                for axis_name, crit_list in cat_map.items():
                    for crit in crit_list:
                        v = scores.get(crit)
                        if v is None:
                            v = norm.get(crit.strip().lower(), 0)
                        axis_pass[(gen_model, dataset, lang, axis_name)].append(int(bool(v)))

    for (m, d, l), xs in sorted(by_key.items()):
        mqs_rows.append({"gen_model": m, "dataset": d, "lang": l,
                         "n": len(xs), "mqs_mean": round(mean(xs), 4),
                         "mqs_std": round(pstdev(xs), 4) if len(xs) > 1 else 0.0})
    for (m, d, l, th), xs in sorted(theme_key.items()):
        if not th:
            continue
        theme_mqs_rows.append({"gen_model": m, "dataset": d, "lang": l, "theme": th,
                               "n": len(xs), "mqs_mean": round(mean(xs), 4)})
    for (m, d, l, ax), xs in sorted(axis_pass.items()):
        axis_rows.append({"gen_model": m, "dataset": d, "lang": l, "axis": ax,
                          "n": len(xs), "pass_rate": round(mean(xs), 4)})

    pd.DataFrame(mqs_rows).to_csv(OUT / f"mqs_per_model__{judge_alias}.csv", index=False)
    pd.DataFrame(theme_mqs_rows).to_csv(OUT / f"mqs_per_theme__{judge_alias}.csv", index=False)
    pd.DataFrame(axis_rows).to_csv(OUT / f"axis_per_model__{judge_alias}.csv", index=False)
    print(f"wrote judge aggregation ({judge_alias}): {len(mqs_rows)} model×ds×lang rows, "
          f"{len(theme_mqs_rows)} theme rows, {len(axis_rows)} axis rows")


def aggregate_metrics():
    rows = []
    by_key: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for path in RUNS_DIR.glob("metrics__*.jsonl"):
        parts = path.stem.split("__")
        if len(parts) != 4:
            continue
        gen_model, dataset, lang = parts[1], parts[2], parts[3]
        for r in checkpoint.jsonl_rows(str(path)):
            for m in ("sacrebleu", "chrf_pp", "rouge_l", "meteor", "bert_f1", "emb_sim"):
                v = r.get(m)
                if v is None:
                    continue
                by_key[(gen_model, dataset, lang)][m].append(float(v))

    for (m, d, l), d_metrics in sorted(by_key.items()):
        row = {"gen_model": m, "dataset": d, "lang": l, "n": max((len(v) for v in d_metrics.values()), default=0)}
        for metric_name, values in d_metrics.items():
            row[metric_name] = round(mean(values), 4) if values else None
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "metrics_per_model.csv", index=False)
    print(f"wrote metrics aggregation: {len(rows)} rows")


def aggregate_validity():
    """Count empty/error responses per (model × dataset × lang × run)."""
    rows = []
    counters: dict[tuple, dict[str, int]] = defaultdict(lambda: {"ok": 0, "empty": 0, "err": 0})
    for path in RUNS_DIR.glob("gen__*.jsonl"):
        parts = path.stem.split("__")
        if len(parts) != 4:
            continue
        gen_model, dataset, lang = parts[1], parts[2], parts[3]
        for r in checkpoint.jsonl_rows(str(path)):
            run = r.get("run")
            key = (gen_model, dataset, lang, run)
            resp = (r.get("response") or "").strip()
            if r.get("error"):
                counters[key]["err"] += 1
            elif not resp:
                counters[key]["empty"] += 1
            else:
                counters[key]["ok"] += 1
    for (m, d, l, run), c in sorted(counters.items()):
        total = c["ok"] + c["empty"] + c["err"]
        rows.append({"gen_model": m, "dataset": d, "lang": l, "run": run,
                     "ok": c["ok"], "empty": c["empty"], "err": c["err"], "total": total,
                     "ok_rate": round(c["ok"] / total, 4) if total else 0.0})
    pd.DataFrame(rows).to_csv(OUT / "valid_response_counts.csv", index=False)
    print(f"wrote validity aggregation: {len(rows)} rows")


def main():
    aggregate_validity()
    for judge in ("gpt_4o_mini", "claude_opus_4_6", "gpt_5_1"):
        if list(RUNS_DIR.glob(f"judge__{judge}__*.jsonl")):
            aggregate_judge(judge)
    aggregate_metrics()


if __name__ == "__main__":
    main()
