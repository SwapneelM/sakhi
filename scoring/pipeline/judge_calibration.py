"""Run a single judge across the pre-built calibration subset.

The subset lives at runs/calibration_subset.jsonl (see calibration_subset.py). For each row we
build the standard zero-shot rubric prompt and route it to one of:

    --provider openrouter   (judge accessed via OpenRouter API, e.g. gpt-5.1, claude-haiku-4-5)
    --provider claude_cli   (judge accessed via local `claude -p`, zero API cost)

Output: runs/judge_cal__<judge_alias>.jsonl (one row per subset row). Resumable via (row_uid).
"""
from __future__ import annotations
import argparse, asyncio, json, os, subprocess, sys, time
from pathlib import Path
from typing import Optional

import aiohttp
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scoring.pipeline import checkpoint
from scoring.pipeline.generate import RateLimiter, RateLimited429
from scoring.pipeline.judge import judge_call as openrouter_judge_call
from scoring.scoring_rubric import RUBRIC_MAP, calc_m, flatten, clean_j

REPO = Path(__file__).resolve().parent.parent.parent
RUNS_DIR = REPO / "runs"

for p in (Path.home() / ".env.openrouter", Path.home() / ".env.gemini"):
    if p.exists():
        load_dotenv(p, override=False)
OR_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip().strip('"\'')


def row_uid(r: dict) -> str:
    return f"{r['gen_model']}|{r['dataset']}|{r['lang']}|{r['run']}|{r['q_idx']}"


def call_claude_cli(prompt: str, model: str = "", *, timeout: int = 180) -> tuple[dict, dict]:
    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)
    sys_msg = "Return valid JSON only. Use the exact rubric text as keys."
    combined = f"{sys_msg}\n\n{prompt}"
    cmd = ["claude", "-p"]
    if model:
        cmd += ["--model", model]
    cmd.append(combined)
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        capture_output=True, text=True, timeout=timeout, check=False, env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"claude CLI rc={proc.returncode}: {(proc.stderr or '')[:300]}")
    txt = (proc.stdout or "").strip()
    parsed = clean_j(txt) or {}
    meta = {"latency_s": round(time.time() - t0, 3), "raw_chars": len(txt)}
    return parsed, meta


async def call_claude_cli_async(prompt: str, model: str = "", *, timeout: int = 180):
    return await asyncio.to_thread(call_claude_cli, prompt, model, timeout=timeout)


async def run(args):
    subset_path = RUNS_DIR / "calibration_subset.jsonl"
    if not subset_path.exists():
        print(f"FATAL: {subset_path} missing; run calibration_subset.py first.", file=sys.stderr)
        return 1
    out_path = RUNS_DIR / f"judge_cal__{args.judge_alias}.jsonl"

    done = set()
    for r in checkpoint.jsonl_rows(str(out_path)):
        if r.get("error"):
            continue
        done.add(r.get("uid"))
    rows = list(checkpoint.jsonl_rows(str(subset_path)))
    pending = [r for r in rows if row_uid(r) not in done]
    print(f"[cal-judge {args.judge_alias}] total={len(rows)} pending={len(pending)}", flush=True)
    if not pending:
        return 0

    sem = asyncio.Semaphore(args.max_concurrent)
    limiter = RateLimiter(args.rpm) if args.rpm > 0 else None

    async def worker(session: Optional[aiohttp.ClientSession], row: dict):
        async with sem:
            uid = row_uid(row)
            theme = (row.get("theme") or "").strip()
            rubric_str = RUBRIC_MAP.get(theme)
            if not rubric_str:
                checkpoint.append(str(out_path), {"uid": uid, "judge_alias": args.judge_alias,
                                                  "error": "unknown_theme"})
                return False
            rubric_list = flatten(rubric_str)
            last_err = None
            for attempt in range(args.retries):
                if limiter:
                    await limiter.wait()
                try:
                    if args.provider == "openrouter":
                        parsed, meta = await openrouter_judge_call(
                            session, args.judge_model,
                            row.get("question_ref") or row.get("question", ""),
                            row.get("reference", ""),
                            row.get("response", ""),
                            rubric_list,
                        )
                    elif args.provider == "claude_cli":
                        prompt = (
                            f"Question:\n{row.get('question_ref') or row.get('question', '')}\n\n"
                            f"Reference:\n{row.get('reference', '')}\n\n"
                            f"Response:\n{row.get('response', '')}\n\n"
                            f"Rubrics:\n" + "\n".join(f"{i+1}. {x}" for i, x in enumerate(rubric_list))
                            + "\n\nReturn JSON: {\"scores\": {\"<Exact Rubric Text>\": 0 or 1}}"
                        )
                        parsed, meta = await call_claude_cli_async(prompt, args.claude_model)
                    else:
                        raise ValueError(args.provider)
                    if not parsed:
                        raise RuntimeError("empty judge output")
                    scores_json, mqs = calc_m(parsed, theme)
                    checkpoint.append(str(out_path), {
                        "uid": uid, "judge_alias": args.judge_alias, "judge_model": args.judge_model,
                        "gen_model": row["gen_model"], "dataset": row["dataset"], "lang": row["lang"],
                        "run": row["run"], "q_idx": row["q_idx"], "theme": theme,
                        "rubric_scores": parsed, "mqs": mqs, "attempt": attempt, **meta,
                    })
                    return True
                except RateLimited429 as e:
                    last_err = f"429: {e}"
                    await asyncio.sleep(e.retry_after)
                except Exception as e:
                    last_err = str(e)[:300]
                    await asyncio.sleep(min(2 ** attempt, 15))
            checkpoint.append(str(out_path), {"uid": uid, "judge_alias": args.judge_alias,
                                              "error": last_err, "attempts": args.retries})
            return False

    if args.provider == "openrouter":
        if not OR_KEY:
            print("FATAL: OPENROUTER_API_KEY not set", file=sys.stderr); return 1
        connector = aiohttp.TCPConnector(limit=args.max_concurrent * 2)
        async with aiohttp.ClientSession(connector=connector) as session:
            results = await asyncio.gather(*[worker(session, r) for r in pending])
    else:
        results = await asyncio.gather(*[worker(None, r) for r in pending])
    ok = sum(1 for r in results if r)
    print(f"[cal-judge {args.judge_alias}] done: ok={ok} fail={len(results) - ok}", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True, choices=["openrouter", "claude_cli"])
    ap.add_argument("--judge-model", required=True, help="e.g. openai/gpt-5.1-20251113 or CLAUDE_CLI_OPUS_4_6 (informational for claude_cli)")
    ap.add_argument("--judge-alias", required=True, help="e.g. gpt_5_1, claude_opus_4_6")
    ap.add_argument("--max-concurrent", type=int, default=4)
    ap.add_argument("--rpm", type=int, default=60)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--claude-model", default="", help="for provider=claude_cli: --model arg to claude (e.g. claude-opus-4-6)")
    args = ap.parse_args()
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
