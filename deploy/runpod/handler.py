"""RunPod serverless handler: run one val-locked training job and return its metrics.

Job input schema (all but `config` optional):
  {
    "config":  {"threshold":0.12,"n_timesteps":64,"orders":[0,1],"hidden":128,
                "scheme":"sqrt","lr":0.004,"epochs":100},
    "seeds":   5,
    "augment": "svdb:data/svdb:0",          # add external DBs to training
    "external":"incart:data/incartdb:1",     # evaluate frozen operating point on external DBs
    "augment_n_cap": 20000
  }

Returns the parsed contents of the harness --out JSON plus the process return code.
"""
import json
import os
import subprocess
import sys

import runpod


def handler(job):
    inp = job.get("input", {}) or {}
    config = inp.get("config")
    if config is None:
        return {"error": "missing 'config' in input"}

    out = "/tmp/result.json"
    if os.path.exists(out):
        os.remove(out)

    cmd = [
        sys.executable,
        "experiments/lock_snn_rr.py",
        "--validate",
        json.dumps(config),
        "--seeds",
        str(inp.get("seeds", 5)),
        "--out",
        out,
    ]
    if inp.get("augment"):
        cmd += ["--augment", inp["augment"]]
    if inp.get("external"):
        cmd += ["--external", inp["external"]]
    if inp.get("augment_n_cap"):
        cmd += ["--augment-n-cap", str(inp["augment_n_cap"])]

    proc = subprocess.run(cmd, cwd="/app", capture_output=True, text=True)
    result = None
    if os.path.exists(out):
        with open(out) as f:
            result = json.load(f)

    return {
        "config": config,
        "result": result,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-3000:],
        "stderr_tail": proc.stderr[-2000:],
    }


runpod.serverless.start({"handler": handler})
