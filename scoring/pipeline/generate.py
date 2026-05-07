"""Resumable generator. Writes per (model, lang) JSONL; resume by skipping completed (run, q_idx) keys.

Usage:
    python -m scoring.pipeline.generate --model gemini-3-flash-preview --provider gemini --lang en [--max-concurrent 6]

Exit codes: 0 on clean completion, 1 on fatal config error. Per-call errors are logged to the JSONL as rows with an "error" field, and rerun will pick them up.
"""
from __future__ import annotations

import argparse, asyncio, os, re, sys, time, signal, subprocess, traceback
from pathlib import Path
from typing import Optional

import aiohttp
import pandas as pd
from dotenv import load_dotenv


class RateLimiter:
    """Async token-bucket limiter: at most `rpm` request-starts per 60s."""
    def __init__(self, rpm: int):
        self.rpm = max(1, int(rpm))
        self.interval = 60.0 / self.rpm
        self._lock = asyncio.Lock()
        self._next_ok_at = 0.0

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            if now < self._next_ok_at:
                await asyncio.sleep(self._next_ok_at - now)
                now = time.monotonic()
            self._next_ok_at = max(now, self._next_ok_at) + self.interval

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scoring.pipeline import checkpoint
from scoring.pipeline.prompts import build_gen_prompt

REPO = Path(__file__).resolve().parent.parent.parent
DATA_NONEXPERT = REPO / "data2" / "sakhi_non_expert_raw_230.csv"
DATA_EXPERT = REPO / "data2" / "sakhi_expert_raw_150.csv"
RUNS_DIR = REPO / "runs"

GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
OR_KEY = os.environ.get("OPENROUTER_API_KEY", "")
for p in [Path.home() / ".env.gemini", Path.home() / ".env.openrouter"]:
    if p.exists():
        load_dotenv(p, override=False)
GEMINI_KEY = GEMINI_KEY or os.environ.get("GEMINI_API_KEY", "")
OR_KEY = OR_KEY or os.environ.get("OPENROUTER_API_KEY", "")
for var in ("GEMINI_KEY", "OR_KEY"):
    val = locals()[var]
    if val and (val.startswith('"') or val.startswith("'")):
        locals()[var] = val.strip("'\"")


DATASET_CONFIG = {
    "expert": {
        "path": DATA_EXPERT,
        "q_cols": {"en": "question", "hi": "question_hi", "mr": "question_mr"},
        "ref_cols": {"en": "ideal_answer", "hi": "ideal_answer_hi", "mr": "ideal_answer_mr"},
        "id_col": "q_id",
    },
    "non_expert": {
        "path": DATA_NONEXPERT,
        "q_cols": {"en": "question", "hi": "questions_hindi", "mr": "questions_marathi"},
        "ref_cols": {"en": "answer", "hi": "answer_hindi", "mr": "answer_marathi"},
        "id_col": None,  # use row index
    },
}


def load_questions(lang: str, dataset: str) -> list[dict]:
    """Load questions for the given dataset track and language."""
    cfg = DATASET_CONFIG[dataset]
    df = pd.read_csv(cfg["path"])
    q_col = cfg["q_cols"][lang]
    rows = []
    for i, r in df.iterrows():
        q = r[q_col]
        if pd.isna(q) or not str(q).strip():
            continue
        q_id = str(r[cfg["id_col"]]) if cfg["id_col"] and cfg["id_col"] in df.columns else str(i)
        rows.append({
            "q_idx": int(i),
            "q_id": q_id,
            "question": str(q).strip(),
            "theme": str(r.get("theme", "")),
        })
    return rows


async def call_gemini(session: aiohttp.ClientSession, model: str, prompt: str, *, timeout: int = 120) -> tuple[str, dict]:
    """Direct Google Generative Language API call. Returns (text, meta).

    Gemini 3 uses thinking tokens that count against maxOutputTokens. For this benchmark we
    set thinkingBudget=0 (Flash, Flash-Lite) or a small budget (Pro, which requires >0),
    and use a 1024-token cap so truncation is not a confound.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_KEY}"
    thinking_budget = 128 if "pro" in model.lower() else 0
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 1024,
            "thinkingConfig": {"thinkingBudget": thinking_budget},
        },
    }
    t0 = time.time()
    async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)) as res:
        if res.status == 429:
            body = await res.text()
            ra = res.headers.get("Retry-After")
            try:
                ra_s = float(ra) if ra else 15.0
            except ValueError:
                ra_s = 15.0
            raise RateLimited429(min(max(ra_s, 1.0), 60.0), body)
        data = await res.json()
        if res.status != 200:
            raise RuntimeError(f"gemini {res.status}: {str(data)[:300]}")
    cand = (data.get("candidates") or [{}])[0]
    parts = ((cand.get("content") or {}).get("parts") or [])
    text = "".join(p.get("text", "") for p in parts).strip()
    usage = data.get("usageMetadata") or {}
    meta = {
        "input_tokens": usage.get("promptTokenCount"),
        "output_tokens": usage.get("candidatesTokenCount"),
        "latency_s": round(time.time() - t0, 3),
        "finish_reason": cand.get("finishReason"),
    }
    return text, meta


class RateLimited429(Exception):
    def __init__(self, retry_after: float, body: str):
        self.retry_after = retry_after
        super().__init__(f"429 rate limited; retry_after={retry_after:.1f}s body={body[:200]}")


OR_REASONING_MODELS = ("gpt-5", "gpt5", "o1", "o3", "grok-4", "gemini-2.5", "gemini-3")


async def call_openrouter(session: aiohttp.ClientSession, model: str, prompt: str, *, timeout: int = 120) -> tuple[str, dict]:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OR_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/SwapneelM/MedicaLLM-Eval",
        "X-Title": "Sakhi Benchmark",
    }
    model_l = model.lower()
    is_reasoner = any(x in model_l for x in OR_REASONING_MODELS)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1.0 if is_reasoner else 0.7,
        "max_tokens": 2000 if is_reasoner else 400,
    }
    if is_reasoner:
        payload["reasoning"] = {"effort": "low", "exclude": True}
    t0 = time.time()
    async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)) as res:
        if res.status == 429:
            body = await res.text()
            ra = res.headers.get("Retry-After") or res.headers.get("X-RateLimit-Reset")
            try:
                ra_s = float(ra) if ra else 10.0
            except ValueError:
                ra_s = 10.0
            raise RateLimited429(min(max(ra_s, 1.0), 60.0), body)
        data = await res.json()
        if res.status != 200:
            raise RuntimeError(f"openrouter {res.status}: {str(data)[:300]}")
    choice = (data.get("choices") or [{}])[0]
    text = ((choice.get("message") or {}).get("content") or "").strip()
    usage = data.get("usage") or {}
    meta = {
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "latency_s": round(time.time() - t0, 3),
        "finish_reason": choice.get("finish_reason"),
    }
    return text, meta


def call_claude_cli(prompt: str, *, timeout: int = 180) -> tuple[str, dict]:
    """Sync: invoke `claude -p` as a subprocess. Zero API cost through Claude Code subscription."""
    t0 = time.time()
    try:
        proc = subprocess.run(
            ["claude", "-p", prompt],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"claude CLI timeout after {timeout}s")
    if proc.returncode != 0:
        raise RuntimeError(f"claude CLI rc={proc.returncode}: {(proc.stderr or '')[:300]}")
    text = (proc.stdout or "").strip()
    meta = {"latency_s": round(time.time() - t0, 3)}
    return text, meta


async def call_claude_cli_async(prompt: str, *, timeout: int = 180) -> tuple[str, dict]:
    return await asyncio.to_thread(call_claude_cli, prompt, timeout=timeout)


async def one_call(provider: str, model: str, session: Optional[aiohttp.ClientSession], prompt: str) -> tuple[str, dict]:
    if provider == "gemini":
        return await call_gemini(session, model, prompt)
    if provider == "openrouter":
        return await call_openrouter(session, model, prompt)
    if provider == "claude_cli":
        return await call_claude_cli_async(prompt)
    raise ValueError(f"unknown provider: {provider}")


async def run(args: argparse.Namespace) -> int:
    provider = args.provider
    model = args.model
    lang = args.lang
    runs = list(range(1, args.num_runs + 1))

    dataset = args.dataset
    model_alias = args.model_alias or model.replace("/", "_")
    jsonl_path = str(RUNS_DIR / f"gen__{model_alias}__{dataset}__{lang}.jsonl")
    key_fields = ("run", "q_idx")
    done = checkpoint.completed_keys(jsonl_path, key_fields)
    print(f"[{model} {dataset} {lang}] resume: {len(done)} already done", flush=True)

    questions = load_questions(lang, dataset)
    if args.limit:
        questions = questions[: args.limit]
    total = len(questions) * len(runs)
    pending = [(r, q) for r in runs for q in questions if (r, q["q_idx"]) not in done]
    print(f"[{model} {dataset} {lang}] total={total} completed={len(done)} pending={len(pending)}", flush=True)

    sem = asyncio.Semaphore(args.max_concurrent)
    limiter = RateLimiter(args.rpm) if args.rpm > 0 else None
    retries = args.retries

    async def worker(session, run_num: int, q: dict):
        prompt = build_gen_prompt(q["question"], lang)
        async with sem:
            last_err = None
            for attempt in range(retries):
                if limiter:
                    await limiter.wait()
                try:
                    text, meta = await one_call(provider, model, session, prompt)
                    checkpoint.append(jsonl_path, {
                        "model": model, "model_alias": model_alias, "provider": provider, "run": run_num,
                        "dataset": dataset, "lang": lang,
                        "q_idx": q["q_idx"], "q_id": q["q_id"], "theme": q["theme"],
                        "question": q["question"], "response": text,
                        "attempt": attempt, **meta,
                    })
                    return True
                except RateLimited429 as e:
                    last_err = f"429: {e}"
                    await asyncio.sleep(e.retry_after)
                except Exception as e:
                    last_err = str(e)[:300]
                    await asyncio.sleep(min(2 ** attempt, 15))
            checkpoint.append(jsonl_path, {
                "model": model, "provider": provider, "run": run_num,
                "dataset": dataset, "lang": lang,
                "q_idx": q["q_idx"], "q_id": q["q_id"], "theme": q["theme"],
                "question": q["question"], "response": None,
                "error": last_err, "attempts": retries,
            })
            return False

    # For claude_cli, no HTTP session needed; pass None.
    if provider == "claude_cli":
        session = None
        results = await asyncio.gather(*[worker(session, r, q) for r, q in pending])
    else:
        timeout_cfg = aiohttp.ClientTimeout(total=180)
        connector = aiohttp.TCPConnector(limit=args.max_concurrent * 2)
        async with aiohttp.ClientSession(timeout=timeout_cfg, connector=connector) as session:
            results = await asyncio.gather(*[worker(session, r, q) for r, q in pending])

    ok = sum(1 for r in results if r)
    print(f"[{model} {lang}] done: ok={ok} fail={len(results) - ok} total_now={len(done) + ok}", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="provider-specific model id (e.g. gemini-3-flash-preview, google/gemma-2-27b-it)")
    ap.add_argument("--model-alias", default="", help="short canonical name (e.g. gpt_5_mini); defaults to model with / replaced by _")
    ap.add_argument("--provider", required=True, choices=["gemini", "openrouter", "claude_cli"])
    ap.add_argument("--lang", required=True, choices=["en", "hi", "mr"])
    ap.add_argument("--dataset", required=True, choices=["expert", "non_expert"], help="expert = sakhi_expert_raw_150.csv (149 Qs); non_expert = sakhi_non_expert_raw_230.csv (231 Qs)")
    ap.add_argument("--num-runs", type=int, default=3)
    ap.add_argument("--max-concurrent", type=int, default=6)
    ap.add_argument("--rpm", type=int, default=0, help="max request-starts per minute (0 = unlimited). OpenRouter paid: 200. Anthropic via OR: 60. Gemini paid: 1000.")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="smoke-test: only first N questions")
    args = ap.parse_args()

    if args.provider == "gemini" and not GEMINI_KEY:
        print("FATAL: GEMINI_API_KEY not set", file=sys.stderr); sys.exit(1)
    if args.provider == "openrouter" and not OR_KEY:
        print("FATAL: OPENROUTER_API_KEY not set", file=sys.stderr); sys.exit(1)

    rc = asyncio.run(run(args))
    sys.exit(rc)


if __name__ == "__main__":
    main()
