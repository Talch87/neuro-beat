#!/usr/bin/env bash
# Build and push the NeuroBeat RunPod image.
# Requires docker and being logged in to your registry (docker login / gh auth).
#
#   deploy/runpod/build_and_push.sh docker.io/<you>/neurobeat-runpod:latest
#
# Alternative with no local Docker: use RunPod's GitHub integration and point the
# serverless endpoint at this repo + deploy/runpod/Dockerfile (see README).
set -euo pipefail
IMAGE="${1:?usage: build_and_push.sh <registry>/<user>/neurobeat-runpod:tag}"
docker build -f deploy/runpod/Dockerfile -t "$IMAGE" .
docker push "$IMAGE"
echo "pushed: $IMAGE"
echo "Use this image when creating your RunPod serverless endpoint."
