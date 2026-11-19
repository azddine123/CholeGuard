#!/usr/bin/env bash
# Download the trained model, thresholds and results published on Hugging Face.
set -euo pipefail
REPO=${HF_REPO:-AZAS123/choleguard}
mkdir -p models docs/results/figures frontend/public
base="https://huggingface.co/$REPO/resolve/main"
curl -fsSL "$base/choleguard.onnx" -o models/choleguard.onnx
curl -fsSL "$base/metrics.json" -o docs/results/metrics.json
cp docs/results/metrics.json frontend/public/metrics.json
echo "OK: models/choleguard.onnx"
