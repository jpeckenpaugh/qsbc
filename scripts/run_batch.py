#!/usr/bin/env python3
"""Bulk-run the cot agent over C3PA sentences.

Loop: sample (per-label) -> fetch INPUT payload -> invoke the `cot` agent via
the opencode CLI -> capture raw response -> POST to /api/reasonings.

Policies:
  * retries with exponential backoff (+jitter) on agent failures
  * throttle/quota detection from opencode stderr -> longer backoff
  * auto-stop when quota throttling persists or too many hard failures
  * resume-safe: duplicates are detected (HTTP 409) and skipped

Usage:
    python3 scripts/run_batch.py --per-label 24 [--label <id>] [--workers 4]
                                 [--max-retries 3] [--backoff-base 5]
                                 [--backoff-max 60] [--stop-after-throttles 5]
                                 [--max-errors 10] [--dry-run]
"""

import argparse
import json
import random
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

API = "http://localhost:8000"
AGENT = "cot"
MODEL = "opencode-go/deepseek-v4-flash"
REPO_ROOT = "."  # opencode run must see .opencode/agents/cot.md
IMAGE = "ghcr.io/anomalyco/opencode"

THROTTLE_KEYWORDS = (
    "429", "402", "529", "rate limit", "rate-limit", "quota", "too many",
    "throttl", "payment required", "exceeded", "busy",
)

DOCKER_BASE = [
    "docker", "run", "--rm",
    "-v", f"{__import__('os').path.expanduser('~')}/.config/opencode:/root/.config/opencode:ro",
    "-v", f"{__import__('os').path.expanduser('~')}/.local/share/opencode/auth.json:/root/.local/share/opencode/auth.json:ro",
    "-v", f"{__import__('os').path.abspath(REPO_ROOT)}:/workdir:ro",
    "-w", "/workdir",
    IMAGE,
    "run", "--agent", AGENT,  # image ENTRYPOINT is already `opencode`
]


def build_cmd(payload_json, runner):
    if runner == "local":
        return ["opencode", "run", "--agent", AGENT, payload_json]
    return DOCKER_BASE + [payload_json]


def api_get(path):
    with urllib.request.urlopen(f"{API}{path}") as r:
        return json.loads(r.read())


def api_post(path, body, retries=3):
    data = json.dumps(body).encode()
    for attempt in range(retries + 1):
        req = urllib.request.Request(
            f"{API}{path}", data=data,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 409:  # already stored -> resume-safe skip
                return {"already_exists": True}
            if e.code >= 500 and attempt < retries:
                time.sleep(min(30.0, 3.0 * (2 ** attempt)) * (0.5 + random.random()))
                continue
            return {"error": f"http {e.code}"}
        except Exception:
            if attempt < retries:
                time.sleep(min(30.0, 3.0 * (2 ** attempt)) * (0.5 + random.random()))
                continue
            raise
    return {"error": "post failed"}


def is_throttled(text):
    t = (text or "").lower()
    return any(k in t for k in THROTTLE_KEYWORDS)


def backoff(attempt, base, cap):
    delay = min(cap, base * (2 ** attempt))
    return delay * (0.5 + random.random())  # jitter


def sample_tasks(per_label, label_id=None, doc_limit=None, debug=False):
    """Yield task dicts {sentence_id, label_id, payload, doc_total}, deduped."""
    if label_id is not None:
        labels = [{"id": label_id}]
    else:
        labels = [l for l in api_get("/api/labels") if l["name"] != "Others"]

    tasks = []
    seen = set()
    for label in labels:
        tries = 0
        got = 0
        while got < per_label and tries < per_label * 10:
            tries += 1
            try:
                url = f"/api/prompt/random?label_id={label['id']}"
                if doc_limit is not None:
                    url += f"&doc_limit={doc_limit}"
                d = api_get(url)
            except Exception:
                continue
            key = d["sentence_id"]
            if key in seen:
                continue
            seen.add(key)
            task = {
                "sentence_id": d["sentence_id"],
                "label_id": d["label_id"],
                "payload": d["payload"],
                "doc_total": d.get("doc_total"),
                "doc_used": len(d["payload"]["Document"]),
            }
            tasks.append(task)
            got += 1
            if debug:
                print(f"  [sample] #{task['sentence_id']} label={task['label_id']} "
                      f"doc={task['doc_used']}/{task['doc_total']}", file=sys.stderr)
        if got < per_label:
            print(f"  [warn] label {label['id']}: only {got}/{per_label} unique samples", file=sys.stderr)
    return tasks


class BatchState:
    def __init__(self, stop_after_throttles, max_errors):
        self.stop_after_throttles = stop_after_throttles
        self.max_errors = max_errors
        self.throttle_count = 0
        self.error_count = 0
        self.lock = threading.Lock()
        self.stop = threading.Event()

    def note_throttle(self):
        with self.lock:
            self.throttle_count += 1
            if self.throttle_count >= self.stop_after_throttles:
                self.stop.set()
                return True
        return False

    def note_error(self):
        with self.lock:
            self.error_count += 1
            if self.error_count >= self.max_errors:
                self.stop.set()
                return True
        return False


def invoke(task, run_id, state, runner, max_retries, backoff_base, backoff_cap, dry_run=False):
    """Run the cot agent for one task with retry/backoff; return a result dict."""
    sentence_id = task["sentence_id"]
    label_id = task["label_id"]
    payload = task["payload"]
    payload_json = json.dumps(payload, ensure_ascii=False)
    if dry_run:
        return {"sentence_id": sentence_id, "label_id": label_id,
                "run_id": run_id, "status": "dry-run", "raw_response": "", "post": None}
    if state.stop.is_set():
        return {"sentence_id": sentence_id, "label_id": label_id,
                "run_id": run_id, "status": "stopped", "raw_response": "", "post": None}

    start = time.time()
    for attempt in range(max_retries + 1):
        try:
            proc = subprocess.run(
                build_cmd(payload_json, runner),
                capture_output=True, text=True, timeout=300, cwd=REPO_ROOT,
            )
        except subprocess.TimeoutExpired:
            proc = None

        if proc is not None and proc.returncode == 0 and proc.stdout.strip():
            raw = proc.stdout.strip()
            post = api_post("/api/reasonings", {
                "sentence_id": sentence_id, "label_id": label_id,
                "run_id": run_id, "model": MODEL, "raw_response": raw,
            })
            return {"sentence_id": sentence_id, "label_id": label_id, "run_id": run_id,
                    "status": "generated", "raw_response": raw, "post": post,
                    "elapsed": round(time.time() - start, 1)}

        err = (proc.stderr if proc else "") + (proc.stdout if proc else "")
        throttled = is_throttled(err)
        if throttled:
            if state.note_throttle():
                return {"sentence_id": sentence_id, "label_id": label_id,
                        "run_id": run_id, "status": "stopped",
                        "raw_response": "", "post": None, "reason": "quota throttled",
                        "elapsed": round(time.time() - start, 1)}
            print(f"  [throttle] #{sentence_id} attempt {attempt + 1}; backing off", file=sys.stderr)
        elif attempt == max_retries:
            state.note_error()
            return {"sentence_id": sentence_id, "label_id": label_id, "run_id": run_id,
                    "status": "error", "raw_response": "", "post": None,
                    "reason": err[:200] or "no output",
                    "elapsed": round(time.time() - start, 1)}

        time.sleep(backoff(attempt, backoff_base, backoff_cap))

    state.note_error()
    return {"sentence_id": sentence_id, "label_id": label_id, "run_id": run_id,
            "status": "error", "raw_response": "", "post": None,
            "reason": "retries exhausted", "elapsed": round(time.time() - start, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-label", type=int, default=24)
    ap.add_argument("--label", type=int, default=None, help="restrict to one label id")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--runner", choices=["docker", "local"], default="local",
                    help="how to invoke opencode: 'local' runs the in-container "
                         "binary directly (default); 'docker' spawns one "
                         "container per job (legacy)")
    ap.add_argument("--batch", default=None, help="run_id (default: batch-<ts>)")
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--backoff-base", type=float, default=5.0)
    ap.add_argument("--backoff-max", type=float, default=60.0)
    ap.add_argument("--stop-after-throttles", type=int, default=5)
    ap.add_argument("--max-errors", type=int, default=10)
    ap.add_argument("--doc-limit", type=int, default=15,
                    help="cap Document array to a centered window of this many "
                         "same-label sentences (target always included)")
    ap.add_argument("--debug", action="store_true", help="verbose, unbuffered progress output")
    ap.add_argument("--dry-run", action="store_true", help="sample + print, don't run agents")
    args = ap.parse_args()

    sys.stdout.reconfigure(line_buffering=True)  # live progress under -T

    run_id = args.batch or f"batch-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    print(f"run_id: {run_id}")

    print("Sampling...")
    tasks = sample_tasks(args.per_label, args.label, args.doc_limit, args.debug)
    print(f"Sampled {len(tasks)} tasks (doc_limit={args.doc_limit})")

    if args.debug:
        try:
            before = api_get("/api/stats")["reasonings"]
            print(f"  reasonings before: {before} -> {before + len(tasks)} after (target)")
        except Exception as e:
            print(f"  [warn] could not fetch reasonings count: {e}", file=sys.stderr)

    if args.dry_run:
        for t in tasks:
            print(f"  #{t['sentence_id']} label={t['label_id']} "
                  f"doc={t['doc_used']}/{t['doc_total']} :: "
                  f"{json.dumps(t['payload'], ensure_ascii=False)[:120]}...")
        return

    state = BatchState(args.stop_after_throttles, args.max_errors)
    print(f"Running with {args.workers} workers "
          f"(retries={args.max_retries}, backoff {args.backoff_base}s->{args.backoff_max}s, "
          f"stop after {args.stop_after_throttles} throttles / {args.max_errors} errors)...")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(invoke, t, run_id, state, args.runner,
                        args.max_retries, args.backoff_base, args.backoff_max)
            for t in tasks
        ]
        done = 0
        for f in futures:
            r = f.result()
            results.append(r)
            done += 1
            extra = f" {r['elapsed']}s" if r.get("elapsed") is not None else ""
            print(f"  [{done}/{len(futures)}] #{r['sentence_id']} {r['status']}{extra} "
                  f"-> {r.get('post', {}).get('parse_status', '-')}")
            if args.debug and r["status"] in ("error", "stopped"):
                print(f"      reason: {r.get('reason', '')[:300]!r}")
            if state.stop.is_set() and done < len(futures):
                print("  [auto-stop] quota/error threshold hit; cancelling remaining jobs", file=sys.stderr)

    from collections import Counter
    print("\n=== SUMMARY ===")
    print("agent status:", dict(Counter(r["status"] for r in results)))
    parsed = [r["post"].get("parse_status") for r in results if r.get("post")]
    print("parse_status:", dict(Counter(parsed)))
    stored = sum(1 for r in results
                 if r.get("post") and (r["post"].get("parse_status") == "ok"
                                       or r["post"].get("already_exists")))
    print(f"stored/verified: {stored}/{len(tasks)}")
    if state.stop.is_set():
        print("[auto-stop] batch stopped early by quota/error policy")
    try:
        now = api_get("/api/stats")["reasonings"]
        print(f"reasonings total now: {now}")
    except Exception:
        pass
    fails = [r for r in results if r["status"] in ("error", "stopped")]
    if fails:
        print(f"not-stored ({len(fails)}):")
        for r in fails[:10]:
            print(f"  #{r['sentence_id']} {r['status']} {r.get('reason', '')[:80]!r}")
    print(f"\nResume later with: python3 scripts/run_batch.py --per-label {args.per_label} "
          f"[--label <id>] --batch {run_id}")


if __name__ == "__main__":
    main()