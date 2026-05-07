"""Dispatch metrics.py over all available gen JSONLs. Keeps concurrency modest to
avoid thrashing CPU on BERTScore and to respect OpenAI embedding rate limits.
"""
from __future__ import annotations
import argparse, asyncio, sys, time, re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
RUNS_DIR = REPO / "runs"
LOG_DIR = RUNS_DIR / "logs"

GEN_RE = re.compile(r"^gen__(?P<model>[^_]+(?:_[^_]+)*?)__(?P<dataset>expert|non_expert)__(?P<lang>en|hi|mr)\.jsonl$")


def discover() -> list[tuple[str, str, str]]:
    out = []
    for p in sorted(RUNS_DIR.glob("gen__*.jsonl")):
        m = GEN_RE.match(p.name)
        if m:
            out.append((m.group("model"), m.group("dataset"), m.group("lang")))
    return out


async def run_child(gm: str, ds: str, lg: str) -> tuple[str, int]:
    tag = f"metrics__{gm}__{ds}__{lg}"
    log = LOG_DIR / f"{tag}.log"
    cmd = [sys.executable, "-m", "scoring.pipeline.metrics", "--gen-model", gm, "--dataset", ds, "--lang", lg]
    with log.open("a", encoding="utf-8") as lf:
        lf.write(f"\n\n===== launched {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n{' '.join(cmd)}\n")
        lf.flush()
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=lf.fileno(), stderr=lf.fileno(), cwd=str(REPO))
        rc = await proc.wait()
    return tag, rc


async def orchestrate(jobs, concurrent: int):
    sem = asyncio.Semaphore(concurrent)
    async def one(j):
        async with sem:
            gm, ds, lg = j
            print(f"[METRICS-START] {gm} {ds} {lg}", flush=True)
            t0 = time.time()
            tag, rc = await run_child(gm, ds, lg)
            dt = time.time() - t0
            print(f"[METRICS-DONE ] {'OK' if rc == 0 else f'FAIL({rc})'} {tag} ({dt:.1f}s)", flush=True)
            return rc
    results = await asyncio.gather(*[one(j) for j in jobs])
    ok = sum(1 for r in results if r == 0)
    print(f"\nMetrics orchestrator complete: {ok}/{len(results)} jobs ok", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrent", type=int, default=2, help="CPU-bound (BERTScore) — keep modest")
    ap.add_argument("--only-model", default="")
    ap.add_argument("--only-dataset", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    jobs = discover()
    if args.only_model:
        jobs = [j for j in jobs if args.only_model in j[0]]
    if args.only_dataset:
        jobs = [j for j in jobs if j[1] == args.only_dataset]
    print(f"Metrics jobs: {len(jobs)}")
    if args.dry_run:
        for j in jobs[:5]:
            print(f"  {j}")
        if len(jobs) > 5:
            print(f"  ...and {len(jobs) - 5} more")
        return
    asyncio.run(orchestrate(jobs, args.concurrent))


if __name__ == "__main__":
    main()
