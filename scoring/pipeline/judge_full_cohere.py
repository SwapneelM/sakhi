"""
Re-judge the entire 13-model × 2-arm × 3-lang Sakhi panel with Cohere
Command A as the primary judge, motivated by the Claude self-family
sycophancy finding (data/judge_family_sycophancy_lift_*.csv).

For each `runs/gen__<gen_alias>__<arm>__<lang>.jsonl` we have, this
script writes a parallel
`runs/judge__cohere_command_a__<gen_alias>__<arm>__<lang>.jsonl`
with one rubric verdict per (run, q_idx) row.

Resumable on (run, q_idx) per output file. Concurrent with asyncio +
ThreadPoolExecutor wrapping Cohere's sync ClientV2 (Cohere's async
client requires v2.x SDK; we keep concurrency through threads).

Rate budget: ~30k calls × 5s per call / 8 concurrent ~= 5-6 hours.
Cost estimate at Command A pricing ($2.50/M input + $10/M output)
and ~1300 tokens per call: ~$165.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scoring.scoring_rubric import RUBRIC_MAP, calc_m, clean_j, flatten

RUNS_DIR = REPO / "runs"
DATA_NONEXPERT = REPO / "data2" / "sakhi_non_expert_raw_230.csv"
DATA_EXPERT = REPO / "data2" / "sakhi_expert_raw_150.csv"

load_dotenv(Path.home() / ".env.cohere")
COHERE_KEY = os.environ.get("COHERE_API_KEY", "")
if not COHERE_KEY:
    sys.exit("COHERE_API_KEY missing")

import cohere

DATASET_CONFIG = {
    "expert": {
        "path": DATA_EXPERT,
        "q_cols": {"en": "question", "hi": "question_hi", "mr": "question_mr"},
        "ref_cols": {"en": "ideal_answer", "hi": "ideal_answer_hi", "mr": "ideal_answer_mr"},
    },
    "non_expert": {
        "path": DATA_NONEXPERT,
        "q_cols": {"en": "question", "hi": "questions_hindi", "mr": "questions_marathi"},
        "ref_cols": {"en": "answer", "hi": "answer_hindi", "mr": "answer_marathi"},
    },
}


def load_references(dataset: str, lang: str) -> dict[int, dict]:
    cfg = DATASET_CONFIG[dataset]
    df = pd.read_csv(cfg["path"])
    q_col = cfg["q_cols"][lang]
    ref_col = cfg["ref_cols"][lang]
    out: dict[int, dict] = {}
    for i, r in df.iterrows():
        theme = str(r.get("theme", "")).strip().strip('"').strip("'").strip()
        out[int(i)] = {
            "question": str(r.get(q_col, "")),
            "reference": str(r.get(ref_col, "")),
            "theme": theme,
        }
    return out


def existing_keys(jsonl_path: Path) -> set[tuple[int, int]]:
    s: set[tuple[int, int]] = set()
    if not jsonl_path.exists():
        return s
    with jsonl_path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("error"):
                continue
            run = int(r.get("run", 1))
            qi = int(r.get("q_idx", -1))
            if qi >= 0:
                s.add((run, qi))
    return s


def cohere_judge(client: cohere.ClientV2, question: str, reference: str,
                 response: str, rubric_list: list[str]) -> tuple[dict, dict]:
    prompt = (
        f"Question:\n{question}\n\nReference:\n{reference}\n\nResponse:\n{response}\n\n"
        f"Rubrics:\n" + "\n".join(f"{i + 1}. {x}" for i, x in enumerate(rubric_list))
        + "\n\nReturn JSON: {\"scores\": {\"<Exact Rubric Text>\": 0 or 1}}"
    )
    t0 = time.time()
    resp = client.chat(
        model="command-a-03-2025",
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
    }
    return parsed, meta


async def judge_one_file(client: cohere.ClientV2, gen_path: Path,
                         executor: ThreadPoolExecutor, concurrency: int) -> None:
    parts = gen_path.stem.split("__")
    if len(parts) != 4 or parts[0] != "gen":
        return
    gen_alias = parts[1]
    arm = parts[2]
    lang = parts[3]
    out_path = RUNS_DIR / f"judge__cohere_command_a__{gen_alias}__{arm}__{lang}.jsonl"
    done = existing_keys(out_path)
    refs = load_references(arm, lang)

    # Collect to-do rows
    rows = []
    with gen_path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("error") or not (r.get("response") or "").strip():
                continue
            run = int(r.get("run", 1))
            qi = int(r.get("q_idx", -1))
            if qi < 0 or (run, qi) in done:
                continue
            rows.append(r)
    if not rows:
        print(f"  [{gen_alias} {arm} {lang}] nothing to do (already {len(done)})", flush=True)
        return
    print(f"  [{gen_alias} {arm} {lang}] {len(rows)} pending (already {len(done)})", flush=True)

    sem = asyncio.Semaphore(concurrency)

    async def worker(row: dict) -> dict:
        run = int(row.get("run", 1))
        qi = int(row.get("q_idx"))
        ref = refs.get(qi, {})
        theme = (ref.get("theme") or row.get("theme") or "").strip().strip('"').strip("'").strip()
        rubric_str = RUBRIC_MAP.get(theme)
        if not rubric_str:
            return {
                "judge_model": "cohere_command_a", "judge_model_id": "command-a-03-2025",
                "gen_model": gen_alias, "dataset": arm, "lang": lang,
                "run": run, "q_idx": qi, "theme": theme,
                "error": "unknown_theme_no_rubric",
            }
        rubric_list = flatten(rubric_str)
        async with sem:
            loop = asyncio.get_running_loop()
            try:
                parsed, meta = await loop.run_in_executor(
                    executor, cohere_judge,
                    client, ref.get("question", ""), ref.get("reference", ""),
                    row.get("response", ""), rubric_list,
                )
                if not parsed:
                    raise RuntimeError("empty judge output")
                _, mqs = calc_m(parsed, theme)
                return {
                    "judge_model": "cohere_command_a", "judge_model_id": "command-a-03-2025",
                    "gen_model": gen_alias, "dataset": arm, "lang": lang,
                    "run": run, "q_idx": qi, "theme": theme,
                    "rubric_scores": parsed, "mqs": mqs, **meta,
                }
            except Exception as e:
                return {
                    "judge_model": "cohere_command_a", "judge_model_id": "command-a-03-2025",
                    "gen_model": gen_alias, "dataset": arm, "lang": lang,
                    "run": run, "q_idx": qi, "theme": theme,
                    "error": str(e)[:300],
                }

    tasks = [asyncio.create_task(worker(r)) for r in rows]
    n_done = 0
    n_err = 0
    t_start = time.time()
    with out_path.open("a") as out:
        for fut in asyncio.as_completed(tasks):
            rec = await fut
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            if rec.get("error"):
                n_err += 1
            else:
                n_done += 1
            if (n_done + n_err) % 20 == 0:
                rate = (n_done + n_err) / max(0.1, time.time() - t_start)
                eta = (len(rows) - n_done - n_err) / max(0.1, rate)
                print(f"    [{gen_alias} {arm} {lang}] {n_done}/{len(rows)} ok, {n_err} err, "
                      f"{rate:.1f}/s, ETA {eta/60:.1f}min", flush=True)
    print(f"  [{gen_alias} {arm} {lang}] DONE: {n_done} ok, {n_err} err -> {out_path.relative_to(REPO)}",
          flush=True)


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--only", default=None,
                    help="optional substring filter on gen_alias to limit dispatch")
    args = ap.parse_args()

    gen_files = sorted(RUNS_DIR.glob("gen__*.jsonl"))
    if args.only:
        gen_files = [p for p in gen_files if args.only in p.stem]
    print(f"Cohere Command A re-judge dispatch")
    print(f"  gen files to process: {len(gen_files)}")
    print(f"  concurrency: {args.concurrency}")

    client = cohere.ClientV2(api_key=COHERE_KEY)
    with ThreadPoolExecutor(max_workers=args.concurrency * 2) as executor:
        # Process files sequentially but each file has its own internal concurrency.
        # That keeps a constant total concurrency level across the run.
        for gp in gen_files:
            await judge_one_file(client, gp, executor, args.concurrency)


if __name__ == "__main__":
    asyncio.run(main())
