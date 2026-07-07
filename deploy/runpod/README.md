# NeuroBeat on RunPod (parallel config sweep)

Run the val-locked training sweep across many RunPod serverless workers, so a
`config x seed` grid finishes in roughly one run's wall-clock instead of one after
another.

**Pick a cheap GPU.** This workload is latency-bound and the model is tiny, so an
RTX A4000 or 3090 runs it about as fast as a 4090 at a fraction of the price. An
A100/H100 is wasted money here.

## Files

| File | Purpose |
|---|---|
| `Dockerfile` | Pinned env (uv + deps) + baked MIT-BIH & svdb data + handler |
| `handler.py` | Serverless handler: runs one config, returns the metrics JSON |
| `sweep.py` | Local dispatcher: fans the grid across the endpoint, collects results |
| `build_and_push.sh` | Build + push the image to your registry (optional) |

## One-time setup

1. RunPod account with a little credit (`$10` is plenty) and a spending limit set.
2. Get the image onto RunPod, either:
   - **GitHub integration (no local Docker):** in RunPod, create a Serverless
     endpoint from this GitHub repo and point it at `deploy/runpod/Dockerfile`.
     RunPod builds it for you.
   - **Local build:** `deploy/runpod/build_and_push.sh docker.io/<you>/neurobeat-runpod:latest`
     (needs `docker login`), then use that image for the endpoint.
3. Endpoint settings: pick a cheap GPU (A4000/3090), set **max workers** to how many
   jobs should run at once (e.g. 10), and a low idle timeout. Copy the **Endpoint ID**.

## Run a sweep

Put credentials in your shell (do **not** paste them into any chat or commit them):

```bash
export RUNPOD_API_KEY=...        # RunPod -> Settings -> API Keys
export RUNPOD_ENDPOINT_ID=...    # from the endpoint you created
pip install runpod
python deploy/runpod/sweep.py
```

Edit the `GRID` in `sweep.py` to choose which configs to sweep. Results stream to
`runs/runpod_sweep.json` as jobs finish.

## Cost

At ~15 min/seed on a cheap GPU (~`$0.20-0.40`/hr): a 5-seed run is well under `$1`,
and a 20-config x 5-seed sweep is roughly `$5-15`. Spot pricing cuts it further.
Parallelism reduces wall-clock, not total GPU-hours.

## Notes

- **INCART** (external test set) is large and left out of the image by default;
  uncomment its line in the `Dockerfile`, or mount a RunPod network volume with the
  data to avoid re-downloading on every worker.
- **No test-set tuning.** Each job runs the val-locked protocol: the operating point
  is fit only on the DS1 holdout and frozen for DS2 and any external database.
- **Vectorizing the SNN time loop** would cut every run's runtime (and cost) several
  fold; it is the highest-leverage optimization and independent of hardware.
