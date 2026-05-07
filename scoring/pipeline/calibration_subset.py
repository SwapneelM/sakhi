"""Build the 500-response stratified calibration subset for multi-judge agreement.

Sampling constraints (balanced across):
  - gen_model (13 models)
  - dataset (expert vs non_expert)
  - lang (en, hi, mr)
  - theme (10 themes)
  - run (1, 2, 3)

Strategy: enumerate all completed generation rows, then for each (dataset, lang, theme) cell,
take approximately floor(500 / cells) responses with proportional-to-population sampling,
rounded to nearest integer. Per-cell rows are then chosen uniformly at random (seeded).

Output: runs/calibration_subset.jsonl — one row per selected response, with fields:
  gen_model, dataset, lang, run, q_idx, theme, question, reference, response
This file is the input to both the Claude 4.6 and GPT-5.1 judge passes.
"""
from __future__ import annotations
import argparse, json, random, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scoring.pipeline import checkpoint
from scoring.pipeline.judge import load_references

REPO = Path(__file__).resolve().parent.parent.parent
RUNS_DIR = REPO / "runs"


def build_subset(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    # Load all completed generation rows from every gen__*.jsonl
    all_rows: list[dict] = []
    for p in sorted(RUNS_DIR.glob("gen__*.jsonl")):
        parts = p.stem.split("__")
        if len(parts) != 4:
            continue
        _, gen_model, dataset, lang = parts
        for r in checkpoint.jsonl_rows(str(p)):
            if r.get("error") or not (r.get("response") or "").strip():
                continue
            r = {**r, "gen_model": gen_model, "dataset": dataset, "lang": lang}
            all_rows.append(r)

    print(f"candidate pool: {len(all_rows)} rows", flush=True)
    if not all_rows:
        return []

    # Stratify by (dataset, lang, theme)
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for r in all_rows:
        cells[(r["dataset"], r["lang"], r.get("theme", "UNKNOWN"))].append(r)

    total_cells = len(cells)
    per_cell_target = max(1, n // total_cells)
    selected: list[dict] = []
    for key, rows in cells.items():
        take = min(per_cell_target, len(rows))
        selected.extend(rng.sample(rows, take))

    # If we're short, fill with additional random rows (without replacement)
    if len(selected) < n:
        remaining = [r for r in all_rows if r not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[: n - len(selected)])

    # If over, trim
    if len(selected) > n:
        rng.shuffle(selected)
        selected = selected[:n]

    # Attach reference answers
    ref_cache: dict[tuple, dict] = {}
    for r in selected:
        key = (r["dataset"], r["lang"])
        if key not in ref_cache:
            ref_cache[key] = load_references(r["dataset"], r["lang"])
        ref = ref_cache[key].get(int(r["q_idx"]), {})
        r["question_ref"] = ref.get("question", r.get("question", ""))
        r["reference"] = ref.get("reference", "")

    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260421)
    ap.add_argument("--out", default=str(RUNS_DIR / "calibration_subset.jsonl"))
    args = ap.parse_args()
    rows = build_subset(args.n, args.seed)
    out = Path(args.out)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # Print stratification summary
    from collections import Counter
    by_cell = Counter((r["dataset"], r["lang"]) for r in rows)
    by_model = Counter(r["gen_model"] for r in rows)
    by_theme = Counter(r.get("theme", "") for r in rows)
    print(f"Wrote {len(rows)} rows to {out}")
    print(f"by (dataset, lang): {dict(by_cell)}")
    print(f"by gen_model: {dict(by_model)}")
    print(f"by theme: {dict(by_theme)}")


if __name__ == "__main__":
    main()
