"""
Run Cohere Command A as a judge over the calibration subset.

This adds a fourth LLM judge alongside GPT-4o-mini, GPT-5.1, and
Claude Opus 4.6. We use the same zero-shot prompt and the same
calibration subset (runs/calibration_subset.jsonl, 500 rows).

Reads:
    runs/calibration_subset.jsonl
Writes:
    runs/judge_cal__cohere_command_a.jsonl

Resumable on row_uid.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scoring.scoring_rubric import RUBRIC_MAP, clean_j

RUNS_DIR = REPO / "runs"
SUBSET = RUNS_DIR / "calibration_subset.jsonl"

load_dotenv(Path.home() / ".env.cohere")
COHERE_KEY = os.environ.get("COHERE_API_KEY", "")
if not COHERE_KEY:
    sys.exit("COHERE_API_KEY missing - source ~/.env.cohere first")

import cohere


def load_references(dataset: str, lang: str) -> dict[int, dict]:
    """Load reference answers indexed by q_idx."""
    import pandas as pd

    if dataset == "expert":
        path = REPO / "data2" / "sakhi_expert_raw_150.csv"
        col = {"en": "ideal_answer", "hi": "ideal_answer_hi", "mr": "ideal_answer_mr"}[lang]
    else:
        path = REPO / "data2" / "sakhi_non_expert_raw_230.csv"
        col = {"en": "answer", "hi": "answer_hindi", "mr": "answer_marathi"}[lang]
    df = pd.read_csv(path)
    return {int(i): {"reference": str(row.get(col, "") or "")} for i, row in df.iterrows()}


def existing_uids(jsonl_path: Path) -> set[str]:
    if not jsonl_path.exists():
        return set()
    out = set()
    with jsonl_path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("error"):
                continue
            uid = r.get("row_uid")
            if uid:
                out.add(uid)
    return out


def row_uid(r: dict) -> str:
    alias = r.get("model_alias") or r.get("gen_model") or r.get("model", "").replace("/", "_")
    return f"{alias}|{r['dataset']}|{r['lang']}|{r['run']}|{r['q_idx']}"


def cohere_judge(client: cohere.ClientV2, model: str, question: str,
                 reference: str, response: str, rubric_list: list[str]) -> tuple[dict, dict]:
    prompt = (
        f"Question:\n{question}\n\nReference:\n{reference}\n\nResponse:\n{response}\n\n"
        f"Rubrics:\n" + "\n".join(f"{i + 1}. {x}" for i, x in enumerate(rubric_list))
        + "\n\nReturn JSON: {\"scores\": {\"<Exact Rubric Text>\": 0 or 1}}"
    )
    t0 = time.time()
    resp = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": "Return valid JSON only. Use the exact rubric text as keys."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=800,
    )
    text = resp.message.content[0].text if resp.message.content else ""
    parsed = clean_j(text) or {}
    usage = getattr(resp, "usage", None)
    meta = {
        "input_tokens": getattr(usage.tokens, "input_tokens", None) if usage else None,
        "output_tokens": getattr(usage.tokens, "output_tokens", None) if usage else None,
        "latency_s": round(time.time() - t0, 3),
        "raw_chars": len(text),
    }
    return parsed, meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--judge-model", default="command-a-03-2025")
    ap.add_argument("--judge-alias", default="cohere_command_a")
    ap.add_argument("--limit", type=int, default=None, help="probe limit")
    args = ap.parse_args()

    out_path = RUNS_DIR / f"judge_cal__{args.judge_alias}.jsonl"
    done = existing_uids(out_path)
    print(f"target: {out_path.relative_to(REPO)}")
    print(f"already done: {len(done)}")

    rows = []
    with SUBSET.open() as f:
        for line in f:
            r = json.loads(line)
            if row_uid(r) in done:
                continue
            rows.append(r)
    if args.limit:
        rows = rows[: args.limit]
    print(f"to do: {len(rows)} of 500")
    if not rows:
        print("nothing to do.")
        return 0

    client = cohere.ClientV2(api_key=COHERE_KEY)
    refs_cache: dict[tuple[str, str], dict[int, dict]] = {}

    n_done = n_err = 0
    t_start = time.time()
    with out_path.open("a") as f:
        for r in rows:
            theme = r.get("theme", "").strip()
            rubric_str = RUBRIC_MAP.get(theme)
            if not rubric_str:
                n_err += 1
                continue
            rubric_list = [s.strip() for s in re.split(r"\n+", rubric_str) if s.strip()]
            key = (r["dataset"], r["lang"])
            if key not in refs_cache:
                refs_cache[key] = load_references(*key)
            ref_row = refs_cache[key].get(int(r["q_idx"]), {})
            reference = ref_row.get("reference", "") or ""

            try:
                parsed, meta = cohere_judge(
                    client, args.judge_model,
                    r["question"], reference, r["response"], rubric_list,
                )
                rec = {
                    "row_uid": row_uid(r),
                    "model_alias": r.get("model_alias") or r.get("gen_model"), "dataset": r["dataset"],
                    "lang": r["lang"], "run": r["run"], "q_idx": r["q_idx"],
                    "q_id": r.get("q_id"), "theme": theme,
                    "judge_model": args.judge_model, "judge_alias": args.judge_alias,
                    "rubric_scores": parsed,
                    "meta": meta,
                    "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
                n_done += 1
            except Exception as e:
                rec = {
                    "row_uid": row_uid(r),
                    "judge_alias": args.judge_alias,
                    "error": str(e)[:300],
                    "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
                n_err += 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if (n_done + n_err) % 25 == 0:
                rate = (n_done + n_err) / max(0.1, time.time() - t_start)
                remain = len(rows) - (n_done + n_err)
                eta = remain / max(0.1, rate)
                print(f"  progress: {n_done}/{len(rows)} ok, {n_err} err, "
                      f"{rate:.1f} req/s, ETA {eta/60:.1f} min", flush=True)
    print(f"\nfinal: {n_done} ok, {n_err} err -> {out_path.relative_to(REPO)}")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
