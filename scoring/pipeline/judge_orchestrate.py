"""Run the GPT-4o-mini rubric judge over all available generation JSONLs.
Scans runs/ for gen__*.jsonl files, skips those already fully judged, and fans out.
Safe to run concurrently with the generation orchestrator — it simply picks up rows as
they become available on each pass.
"""
from __future__ import annotations
import argparse, asyncio, os, re, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
RUNS_DIR = REPO / "runs"
LOG_DIR = RUNS_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

GEN_RE = re.compile(r"^gen__(?P<model>[^_]+(?:_[^_]+)*?)__(?P<dataset>expert|non_expert)__(?P<lang>en|hi|mr)\.jsonl$")


def discover_jobs() -> list[tuple[str, str, str]]:
    """Return list of (gen_model_alias, dataset, lang) to judge."""
    out = []
    for p in sorted(RUNS_DIR.glob("gen__*.jsonl")):
        m = GEN_RE.match(p.name)
        if not m:
            continue
        out.append((m.group("model"), m.group("dataset"), m.group("lang")))
    return out


async def run_child(gen_model: str, dataset: str, lang: str, judge_id: str, judge_alias: str,
                    max_concurrent: int, rpm: int) -> tuple[str, int]:
    tag = f"judge_{judge_alias}__{gen_model}__{dataset}__{lang}"
    log = LOG_DIR / f"{tag}.log"
    cmd = [
        sys.executable, "-m", "scoring.pipeline.judge",
        "--gen-model", gen_model,
        "--dataset", dataset,
        "--lang", lang,
        "--judge", judge_id,
        "--judge-alias", judge_alias,
        "--max-concurrent", str(max_concurrent),
        "--rpm", str(rpm),
    ]
    with log.open("a", encoding="utf-8") as lf:
        lf.write(f"\n\n===== launched {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n{' '.join(cmd)}\n\n")
        lf.flush()
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=lf.fileno(), stderr=lf.fileno(), cwd=str(REPO))
        rc = await proc.wait()
    return tag, rc


async def orchestrate(jobs, judge_id: str, judge_alias: str, concurrent_children: int, per_child_concurrent: int, rpm: int):
    sem = asyncio.Semaphore(concurrent_children)

    async def one(j):
        async with sem:
            gm, ds, lg = j
            print(f"[JUDGE-START] {judge_alias} / {gm} {ds} {lg}", flush=True)
            t0 = time.time()
            tag, rc = await run_child(gm, ds, lg, judge_id, judge_alias, per_child_concurrent, rpm)
            dt = time.time() - t0
            status = "OK" if rc == 0 else f"FAIL(rc={rc})"
            print(f"[JUDGE-DONE ] {status} {tag}  ({dt:.1f}s)", flush=True)
            return rc

    results = await asyncio.gather(*[one(j) for j in jobs])
    ok = sum(1 for r in results if r == 0)
    print(f"\nJudge orchestrator complete: {ok}/{len(results)} jobs ok", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="openai/gpt-4o-mini")
    ap.add_argument("--judge-alias", default="gpt_4o_mini")
    ap.add_argument("--concurrent-children", type=int, default=4)
    ap.add_argument("--per-child-concurrent", type=int, default=6)
    ap.add_argument("--rpm", type=int, default=150)
    ap.add_argument("--only-model", default="")
    ap.add_argument("--only-dataset", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    jobs = discover_jobs()
    if args.only_model:
        jobs = [j for j in jobs if args.only_model in j[0]]
    if args.only_dataset:
        jobs = [j for j in jobs if j[1] == args.only_dataset]
    print(f"Judge jobs: {len(jobs)} (judge={args.judge_alias})")
    for j in jobs[:5]:
        print(f"  {j}")
    if len(jobs) > 5:
        print(f"  ... and {len(jobs) - 5} more")
    if args.dry_run:
        return
    asyncio.run(orchestrate(jobs, args.judge, args.judge_alias, args.concurrent_children, args.per_child_concurrent, args.rpm))


if __name__ == "__main__":
    main()
