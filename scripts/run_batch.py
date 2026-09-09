#!/usr/bin/env python3
"""Bulk-run the cot agent over C3PA sentences.

Loop: sample (per-label) -> fetch INPUT payload -> invoke the `cot` agent via
the opencode CLI -> capture raw response -> POST to /api/reasonings.

Usage:
    python3 scripts/run_batch.py --per-label 24 [--label <id>] [--workers 4]
                                 [--dry-run]
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

API = "http://localhost:8000"
AGENT = "cot"
MODEL = "opencode-go/deepseek-v4-flash"
REPO_ROOT = "."  # opencode run must see .opencode/agents/cot.md
IMAGE = "ghcr.io/anomalyco/opencode"

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


def api_post(path, body):
    req = urllib.request.Request(
        f"{API}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def sample_tasks(per_label, label_id=None):
    """Yield (sentence_id, label_id, payload) for one batch, deduped."""
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
                d = api_get(f"/api/prompt/random?label_id={label['id']}")
            except Exception:
                continue
            key = d["sentence_id"]
            if key in seen:
                continue
            seen.add(key)
            tasks.append((d["sentence_id"], d["label_id"], d["payload"]))
            got += 1
        if got < per_label:
            print(f"  [warn] label {label['id']}: only {got}/{per_label} unique samples", file=sys.stderr)
    return tasks


def invoke(sentence_id, label_id, payload, run_id, runner="docker", dry_run=False):
    """Run the cot agent for one payload; return a result dict."""
    payload_json = json.dumps(payload, ensure_ascii=False)
    if dry_run:
        return {
            "sentence_id": sentence_id, "label_id": label_id,
            "run_id": run_id, "status": "dry-run", "raw_response": "",
            "post": None,
        }
    try:
        proc = subprocess.run(
            build_cmd(payload_json, runner),
            capture_output=True, text=True, timeout=300, cwd=REPO_ROOT,
        )
    except subprocess.TimeoutExpired:
        return {
            "sentence_id": sentence_id, "label_id": label_id, "run_id": run_id,
            "status": "error", "raw_response": "", "post": None,
        }
    raw = proc.stdout.strip()
    if proc.returncode != 0:
        status = "error"
    elif not raw:
        status = "empty"
    else:
        status = "generated"
    try:
        post = api_post("/api/reasonings", {
            "sentence_id": sentence_id,
            "label_id": label_id,
            "run_id": run_id,
            "model": MODEL,
            "raw_response": raw,
        })
    except Exception as e:
        post = {"error": str(e)}
    return {
        "sentence_id": sentence_id, "label_id": label_id, "run_id": run_id,
        "status": status, "raw_response": raw, "post": post,
    }


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
    ap.add_argument("--dry-run", action="store_true", help="sample + print, don't run agents")
    args = ap.parse_args()

    run_id = args.batch or f"batch-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    print(f"run_id: {run_id}")

    print("Sampling...")
    tasks = sample_tasks(args.per_label, args.label)
    print(f"Sampled {len(tasks)} tasks")

    if args.dry_run:
        for sid, lid, payload in tasks:
            print(f"  #{sid} label={lid} :: {json.dumps(payload, ensure_ascii=False)[:120]}...")
        return

    print(f"Running with {args.workers} workers...")
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(invoke, sid, lid, payload, run_id, args.runner)
            for sid, lid, payload in tasks
        ]
        done = 0
        for f in futures:
            r = f.result()
            results.append(r)
            done += 1
            print(f"  [{done}/{len(futures)}] #{r['sentence_id']} {r['status']} "
                  f"-> {r.get('post', {}).get('parse_status')}")

    from collections import Counter
    print("\n=== SUMMARY ===")
    print("agent status:", dict(Counter(r["status"] for r in results)))
    parsed = [r["post"].get("parse_status") for r in results if r.get("post")]
    print("parse_status:", dict(Counter(parsed)))
    fails = [r for r in results if r["status"] != "generated" or (r.get("post") and r["post"].get("parse_status") != "ok")]
    if fails:
        print(f"notes ({len(fails)}):")
        for r in fails[:10]:
            print(f"  #{r['sentence_id']} {r['status']} raw={r['raw_response'][:80]!r}")


if __name__ == "__main__":
    main()