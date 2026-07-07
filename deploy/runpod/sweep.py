"""Fan a config x seed grid across a RunPod serverless endpoint and collect results.

Credentials are read from the environment (never hardcode, never paste in chat):
  RUNPOD_API_KEY       your RunPod API key
  RUNPOD_ENDPOINT_ID   the serverless endpoint running the NeuroBeat image

Usage:
  pip install runpod
  export RUNPOD_API_KEY=...  RUNPOD_ENDPOINT_ID=...
  python deploy/runpod/sweep.py

Each GRID entry becomes one job (with its own seeds), dispatched concurrently; the
endpoint's max-workers setting controls how many run at once. Results stream to
runs/runpod_sweep.json as jobs finish.
"""
import json
import os
import time
from pathlib import Path

import runpod

runpod.api_key = os.environ["RUNPOD_API_KEY"]
ENDPOINT = runpod.Endpoint(os.environ["RUNPOD_ENDPOINT_ID"])
OUT = Path("runs/runpod_sweep.json")

# --- edit this grid: one dict per training job --------------------------------
GRID = [
    {"threshold": 0.12, "n_timesteps": 64, "orders": [0, 1], "hidden": 128,
     "scheme": "sqrt", "lr": 0.004, "epochs": 100},
    # {"threshold": 0.10, "n_timesteps": 64, "orders": [0, 1], "hidden": 128, ...},
    # add more configs to sweep here ...
]
# options applied to every job (add "external":"incart:data/incartdb:1" if INCART is baked in)
COMMON = {"seeds": 5, "augment": "svdb:data/svdb:0", "augment_n_cap": 20000}
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}
# ------------------------------------------------------------------------------


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    jobs = [(c, ENDPOINT.run({"config": c, **COMMON})) for c in GRID]
    print(f"submitted {len(jobs)} jobs to endpoint {os.environ['RUNPOD_ENDPOINT_ID']}", flush=True)

    results = []
    for c, job in jobs:
        status = job.status()
        while status not in TERMINAL:
            time.sleep(15)
            status = job.status()
        output = job.output() if status == "COMPLETED" else None
        results.append({"config": c, "status": status, "output": output})
        OUT.write_text(json.dumps(results, indent=2))
        print(f"[{status}] {c}", flush=True)

    print(f"done -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
