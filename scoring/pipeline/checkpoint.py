"""Resumable JSONL checkpoint. One file per (stage, model, lang). Each line is one call."""
import json, os, time
from pathlib import Path
from threading import Lock

_LOCKS: dict[str, Lock] = {}
_LOCK_REGISTRY = Lock()

def _lock_for(path: str) -> Lock:
    with _LOCK_REGISTRY:
        if path not in _LOCKS:
            _LOCKS[path] = Lock()
        return _LOCKS[path]

def append(path: str, row: dict) -> None:
    """Append a single JSON row to the file. Caller supplies all fields; we add ts."""
    row = {**row, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    lock = _lock_for(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def completed_keys(path: str, key_fields: tuple[str, ...]) -> set[tuple]:
    """Read the JSONL and return the set of already-completed keys (tuples of key_fields).

    A row counts as complete if it has no 'error' field set to a truthy value.
    """
    if not os.path.exists(path):
        return set()
    done: set[tuple] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("error"):
                continue
            try:
                key = tuple(row[k] for k in key_fields)
            except KeyError:
                continue
            done.add(key)
    return done

def jsonl_rows(path: str):
    """Iterate valid JSON rows from a JSONL file (skips broken lines)."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
