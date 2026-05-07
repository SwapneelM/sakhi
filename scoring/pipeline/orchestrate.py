"""Launch the generation matrix for all models × {expert, non_expert} × {en, hi, mr}.
Resumes safely: each child invocation of `generate.py` skips already-done (run, q_idx) pairs.

Per-provider concurrency caps keep us within rate-limit budgets. Each child's stdout/stderr
goes to its own log file under runs/logs/.
"""
from __future__ import annotations
import argparse, asyncio, os, signal, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
RUNS_DIR = REPO / "runs"
LOG_DIR = RUNS_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Panel of models: (model_id, provider, kind)
# kind: "new" = never generated before, runs on both datasets
#       "existing_only_expert" = already in non_expert via migrate_existing, needs expert-track generation
MODELS = [
    # (model_id, provider, kind, canonical_alias)
    # NEW models (need generation on BOTH datasets)
    ("gemini-3-flash-preview",       "gemini",     "new", "gemini_3_flash"),
    ("gemini-3-pro-preview",         "gemini",     "new", "gemini_3_pro"),
    ("gemini-3.1-flash-lite-preview","gemini",     "new", "gemini_3_1_flash_lite"),
    ("google/gemma-3-27b-it",        "openrouter", "new", "gemma_3_27b"),
    ("anthropic/claude-haiku-4-5",   "openrouter", "new", "claude_haiku_4_5"),
    ("CLAUDE_CLI_OPUS_4_7",          "claude_cli", "new", "claude_opus_4_7"),
    # EXISTING models (already in non_expert via migration; need expert-track generation only)
    ("openai/gpt-5-mini",            "openrouter", "existing_only_expert", "gpt_5_mini"),
    ("openai/gpt-4o-mini",           "openrouter", "existing_only_expert", "gpt_4o_mini"),
    ("cohere/command-a",             "openrouter", "existing_only_expert", "cohere_command_a"),
    ("meta-llama/llama-3.3-70b-instruct", "openrouter", "existing_only_expert", "llama_3_3_70b"),
    ("meta-llama/llama-4-maverick",  "openrouter", "existing_only_expert", "llama_4_maverick"),
    ("cohere/aya-expanse-32b",       "openrouter", "existing_only_expert", "aya_expanse"),
    # MedGemma 4B/27B: no OpenRouter endpoint; kept in non_expert via migration only.
    # Expert-track coverage for MedGemma is deferred (noted as dataset-availability limitation).
]

PROVIDER_LIMITS = {
    "gemini":      {"concurrent_children": 3, "per_child_concurrent": 4, "per_child_rpm": 60},
    "openrouter":  {"concurrent_children": 3, "per_child_concurrent": 4, "per_child_rpm": 120},
    "claude_cli":  {"concurrent_children": 1, "per_child_concurrent": 4, "per_child_rpm": 0},
}


def jobs_for(dataset_filter: set[str]) -> list[tuple]:
    """Return list of (model, provider, dataset, lang, alias) tuples to run."""
    out = []
    for model, provider, kind, alias in MODELS:
        datasets = []
        if kind == "new":
            datasets = ["expert", "non_expert"]
        elif kind == "existing_only_expert":
            datasets = ["expert"]
        for ds in datasets:
            if ds not in dataset_filter:
                continue
            for lang in ("en", "hi", "mr"):
                out.append((model, provider, ds, lang, alias))
    return out


async def run_child(model: str, provider: str, dataset: str, lang: str, alias: str, *, num_runs: int) -> tuple[str, int]:
    """Launch `generate.py` as a subprocess. Returns (tag, returncode)."""
    limits = PROVIDER_LIMITS[provider]
    tag = f"{alias}__{dataset}__{lang}"
    log_path = LOG_DIR / f"{tag}.log"
    cmd = [
        sys.executable, "-m", "scoring.pipeline.generate",
        "--model", model,
        "--model-alias", alias,
        "--provider", provider,
        "--lang", lang,
        "--dataset", dataset,
        "--num-runs", str(num_runs),
        "--max-concurrent", str(limits["per_child_concurrent"]),
        "--rpm", str(limits["per_child_rpm"]),
    ]
    env = os.environ.copy()
    if provider == "claude_cli":
        # Let Claude Code subscription handle auth; env var would force external API key.
        env.pop("ANTHROPIC_API_KEY", None)
    with log_path.open("a", encoding="utf-8") as lf:
        lf.write(f"\n\n===== launched {time.strftime('%Y-%m-%d %H:%M:%S')} =====\ncmd: {' '.join(cmd)}\n\n")
        lf.flush()
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=lf.fileno(),
            stderr=lf.fileno(),
            cwd=str(REPO),
            env=env,
        )
        rc = await proc.wait()
    return tag, rc


async def orchestrate(jobs: list[tuple], num_runs: int):
    by_provider: dict[str, list] = {}
    for j in jobs:
        by_provider.setdefault(j[1], []).append(j)

    async def per_provider_worker(provider: str, provider_jobs: list):
        sem = asyncio.Semaphore(PROVIDER_LIMITS[provider]["concurrent_children"])
        async def one(j):
            async with sem:
                model, _prov, ds, lang, alias = j
                print(f"[START] {provider} {alias} {ds} {lang}", flush=True)
                t0 = time.time()
                tag, rc = await run_child(model, provider, ds, lang, alias, num_runs=num_runs)
                dt = time.time() - t0
                status = "OK" if rc == 0 else f"FAIL(rc={rc})"
                print(f"[DONE ] {status} {tag}  ({dt:.1f}s)", flush=True)
                return (tag, rc)
        return await asyncio.gather(*[one(j) for j in provider_jobs])

    results = await asyncio.gather(*[per_provider_worker(p, js) for p, js in by_provider.items()])
    flat = [r for group in results for r in group]
    ok = sum(1 for _, rc in flat if rc == 0)
    print(f"\nOrchestrator complete: {ok}/{len(flat)} jobs ok", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["expert", "non_expert"], choices=["expert", "non_expert"])
    ap.add_argument("--num-runs", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only-model", default="", help="if set, only run this model id (substring match)")
    ap.add_argument("--only-provider", default="", help="if set, only run this provider")
    args = ap.parse_args()

    jobs = jobs_for(set(args.datasets))
    if args.only_model:
        jobs = [j for j in jobs if args.only_model in j[0]]
    if args.only_provider:
        jobs = [j for j in jobs if j[1] == args.only_provider]

    print(f"Total jobs: {len(jobs)}  (num_runs={args.num_runs}, datasets={args.datasets})")
    for j in jobs:
        print(f"  {j}")
    if args.dry_run:
        return
    asyncio.run(orchestrate(jobs, num_runs=args.num_runs))


if __name__ == "__main__":
    main()
