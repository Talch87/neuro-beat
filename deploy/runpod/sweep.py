"""Fan a config grid across a RunPod serverless endpoint and collect results.

Uses the REST /run endpoint with an executionTimeout policy, because serverless
kills long jobs under its default timeout (a 90-min job fails; a ~27-min single-seed
job with a 60-min policy completes). Keep each job short (few seeds) and the policy
generous.

Credentials are read from the environment or deploy/runpod/.env (never hardcode,
never paste in chat):
  RUNPOD_API_KEY       your RunPod API key
  RUNPOD_ENDPOINT_ID   the serverless endpoint running the NeuroBeat image

Usage:
  python deploy/runpod/sweep.py     # loads .env automatically if present
"""
import json
import os
import time
import urllib.request
from pathlib import Path

_envf = Path("deploy/runpod/.env")
if _envf.exists():
    for _line in _envf.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

API = os.environ["RUNPOD_API_KEY"]
EID = os.environ["RUNPOD_ENDPOINT_ID"]
BASE = f"https://api.runpod.ai/v2/{EID}"
HDRS = {"Authorization": f"Bearer {API}", "Content-Type": "application/json"}
OUT = Path("runs/runpod_sweep.json")
EXEC_TIMEOUT_MS = 3600000  # 60 min per job; raise if you increase seeds-per-job
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}

# --- edit this grid: one dict per training job -------------------------------
# SVEB-push sweep (all svdb-augmented), 1 seed each for a fast relative read.
GRID = [
    {"threshold": 0.12, "n_timesteps": 64, "orders": [0, 1], "hidden": 128,
     "scheme": "sqrt", "lr": 0.004, "epochs": 100},   # F4 reference
    {"threshold": 0.12, "n_timesteps": 64, "orders": [0, 1], "hidden": 192,
     "scheme": "sqrt", "lr": 0.004, "epochs": 100},   # more capacity (~21k SynOps)
    {"threshold": 0.12, "n_timesteps": 96, "orders": [0, 1], "hidden": 128,
     "scheme": "sqrt", "lr": 0.004, "epochs": 100},   # more time resolution
    {"threshold": 0.10, "n_timesteps": 64, "orders": [0, 1], "hidden": 128,
     "scheme": "sqrt", "lr": 0.004, "epochs": 100},   # more input detail
]
COMMON = {"seeds": 1, "augment": "svdb:data/svdb:0", "augment_n_cap": 20000}
# ------------------------------------------------------------------------------


def _post(path, payload):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(), headers=HDRS, method="POST"
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def _get(path):
    req = urllib.request.Request(BASE + path, headers=HDRS)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    jobs = []
    for cfg in GRID:
        sub = _post("/run", {"input": {"config": cfg, **COMMON},
                             "policy": {"executionTimeout": EXEC_TIMEOUT_MS}})
        jobs.append((cfg, sub["id"]))
        print(f"submitted {sub['id']}  {cfg}", flush=True)

    results = []
    for cfg, jid in jobs:
        rec = _get(f"/status/{jid}")
        while rec.get("status") not in TERMINAL:
            time.sleep(15)
            rec = _get(f"/status/{jid}")
        results.append({"config": cfg, "status": rec.get("status"),
                        "output": rec.get("output"), "error": rec.get("error")})
        OUT.write_text(json.dumps(results, indent=2))
        print(f"[{rec.get('status')}] {cfg}", flush=True)

    print(f"done -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
