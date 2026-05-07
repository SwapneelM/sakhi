"""Resumable rubric judge. Reads generation JSONLs, scores each response using the Sakhi rubric,
writes per-judgment JSONL. Uses the unchanged zero-shot rubric prompt from the existing
`scoring_rubric.py` to preserve comparability with the paper.

Usage:
    python -m scoring.pipeline.judge --gen-model gemini_3_flash --dataset expert --lang en \
        --judge openai/gpt-4o-mini --judge-alias gpt_4o_mini [--max-concurrent 4 --rpm 180]
"""
from __future__ import annotations
import argparse, asyncio, json, os, sys, time
from pathlib import Path

import aiohttp
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scoring.pipeline import checkpoint
from scoring.pipeline.generate import DATASET_CONFIG, RateLimiter, RateLimited429
# Reuse rubric map + MQS computation from the existing scorer to preserve exactness.
from scoring.scoring_rubric import RUBRIC_MAP, AXIS_MAP, clean_j, calc_m, flatten

REPO = Path(__file__).resolve().parent.parent.parent
RUNS_DIR = REPO / "runs"

for p in [Path.home() / ".env.openrouter", Path.home() / ".env.gemini"]:
    if p.exists():
        load_dotenv(p, override=False)
OR_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip().strip('"\'')


def load_references(dataset: str, lang: str) -> dict[int, dict]:
    """Return {q_idx: {question, reference_answer, theme}}.

    Strips stray surrounding quotes from theme: 3 themes in the expert CSV ship as
    e.g. '"Nutrition, Diet & Supplementation"' (literal quotes), which RUBRIC_MAP
    does not match on.
    """
    cfg = DATASET_CONFIG[dataset]
    df = pd.read_csv(cfg["path"])
    q_col = cfg["q_cols"][lang]
    ref_col = cfg["ref_cols"][lang]
    out = {}
    for i, r in df.iterrows():
        theme = str(r.get("theme", "")).strip().strip('"').strip("'").strip()
        out[int(i)] = {
            "question": str(r.get(q_col, "")),
            "reference": str(r.get(ref_col, "")),
            "theme": theme,
        }
    return out


async def judge_call(session: aiohttp.ClientSession, judge_model: str, question: str,
                     reference: str, response: str, rubric_list: list[str],
                     *, timeout: int = 60) -> tuple[dict, dict]:
    prompt = (
        f"Question:\n{question}\n\nReference:\n{reference}\n\nResponse:\n{response}\n\n"
        f"Rubrics:\n" + "\n".join(f"{i + 1}. {x}" for i, x in enumerate(rubric_list))
        + "\n\nReturn JSON: {\"scores\": {\"<Exact Rubric Text>\": 0 or 1}}"
    )
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OR_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/SwapneelM/MedicaLLM-Eval",
        "X-Title": "Sakhi Judge",
    }
    judge_l = judge_model.lower()
    is_reasoner = any(x in judge_l for x in ("gpt-5", "gpt5", "o1", "o3"))
    payload = {
        "model": judge_model,
        "messages": [
            {"role": "system", "content": "Return valid JSON only. Use the exact rubric text as keys."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 1.0 if is_reasoner else 0.0,
        "max_tokens": 4000 if is_reasoner else 800,
    }
    if is_reasoner:
        payload["reasoning"] = {"effort": "low", "exclude": True}
    t0 = time.time()
    async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)) as res:
        if res.status == 429:
            body = await res.text()
            ra = res.headers.get("Retry-After")
            try:
                ra_s = float(ra) if ra else 5.0
            except ValueError:
                ra_s = 5.0
            raise RateLimited429(min(max(ra_s, 1.0), 60.0), body)
        data = await res.json()
        if res.status != 200:
            raise RuntimeError(f"judge {res.status}: {str(data)[:300]}")
    txt = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
    parsed = clean_j(txt) or {}
    usage = data.get("usage") or {}
    meta = {
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "latency_s": round(time.time() - t0, 3),
        "raw_chars": len(txt),
    }
    return parsed, meta


async def run(args):
    gen_path = RUNS_DIR / f"gen__{args.gen_model}__{args.dataset}__{args.lang}.jsonl"
    out_path = RUNS_DIR / f"judge__{args.judge_alias}__{args.gen_model}__{args.dataset}__{args.lang}.jsonl"
    if not gen_path.exists():
        print(f"FATAL: generation file not found: {gen_path}", file=sys.stderr)
        return 1

    key_fields = ("run", "q_idx")
    done = checkpoint.completed_keys(str(out_path), key_fields)
    print(f"[judge {args.judge_alias} {args.gen_model} {args.dataset} {args.lang}] resume: {len(done)} already scored", flush=True)

    refs = load_references(args.dataset, args.lang)
    # Load all generation rows
    rows = list(checkpoint.jsonl_rows(str(gen_path)))
    # Keep only valid, de-duplicate latest per (run, q_idx)
    latest: dict[tuple, dict] = {}
    for r in rows:
        if r.get("error") or not (r.get("response") or "").strip():
            continue
        k = (r.get("run"), r.get("q_idx"))
        latest[k] = r
    pending = [r for k, r in latest.items() if k not in done]
    print(f"[judge {args.judge_alias} {args.gen_model} {args.dataset} {args.lang}] total={len(latest)} pending={len(pending)}", flush=True)

    sem = asyncio.Semaphore(args.max_concurrent)
    limiter = RateLimiter(args.rpm) if args.rpm > 0 else None

    async def worker(session, row: dict):
        async with sem:
            q_idx = int(row.get("q_idx"))
            ref = refs.get(q_idx, {})
            theme = ref.get("theme") or row.get("theme") or ""
            rubric_str = RUBRIC_MAP.get(theme.strip())
            if not rubric_str:
                checkpoint.append(str(out_path), {
                    "judge_model": args.judge_alias, "judge_model_id": args.judge,
                    "gen_model": args.gen_model, "dataset": args.dataset, "lang": args.lang,
                    "run": row.get("run"), "q_idx": q_idx, "theme": theme,
                    "error": "unknown_theme_no_rubric",
                })
                return False
            rubric_list = flatten(rubric_str)
            last_err = None
            for attempt in range(args.retries):
                if limiter:
                    await limiter.wait()
                try:
                    parsed, meta = await judge_call(
                        session, args.judge, ref.get("question", ""), ref.get("reference", ""),
                        row.get("response", ""), rubric_list,
                    )
                    if not parsed:
                        raise RuntimeError("empty judge output")
                    scores_json, mqs = calc_m(parsed, theme)
                    checkpoint.append(str(out_path), {
                        "judge_model": args.judge_alias, "judge_model_id": args.judge,
                        "gen_model": args.gen_model, "dataset": args.dataset, "lang": args.lang,
                        "run": row.get("run"), "q_idx": q_idx, "theme": theme,
                        "rubric_scores": parsed, "mqs": mqs,
                        "attempt": attempt, **meta,
                    })
                    return True
                except RateLimited429 as e:
                    last_err = f"429: {e}"
                    await asyncio.sleep(e.retry_after)
                except Exception as e:
                    last_err = str(e)[:300]
                    await asyncio.sleep(min(2 ** attempt, 15))
            checkpoint.append(str(out_path), {
                "judge_model": args.judge_alias, "judge_model_id": args.judge,
                "gen_model": args.gen_model, "dataset": args.dataset, "lang": args.lang,
                "run": row.get("run"), "q_idx": q_idx, "theme": theme,
                "error": last_err, "attempts": args.retries,
            })
            return False

    connector = aiohttp.TCPConnector(limit=args.max_concurrent * 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        results = await asyncio.gather(*[worker(session, r) for r in pending])
    ok = sum(1 for r in results if r)
    print(f"[judge {args.judge_alias} {args.gen_model} {args.dataset} {args.lang}] done: ok={ok} fail={len(results) - ok}", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-model", required=True, help="alias as in gen__<alias>__...jsonl")
    ap.add_argument("--dataset", required=True, choices=["expert", "non_expert"])
    ap.add_argument("--lang", required=True, choices=["en", "hi", "mr"])
    ap.add_argument("--judge", required=True, help="OpenRouter model id of the judge")
    ap.add_argument("--judge-alias", required=True)
    ap.add_argument("--max-concurrent", type=int, default=6)
    ap.add_argument("--rpm", type=int, default=180)
    ap.add_argument("--retries", type=int, default=4)
    args = ap.parse_args()

    if not OR_KEY:
        print("FATAL: OPENROUTER_API_KEY not set", file=sys.stderr); sys.exit(1)
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
