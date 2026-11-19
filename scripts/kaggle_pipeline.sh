#!/usr/bin/env bash
# Full training pipeline (run inside a Kaggle GPU notebook from the repo root).
set -euo pipefail
DATA=${DATA:-/kaggle/tmp/data}
OUT=${OUT:-/kaggle/working/out}
mkdir -p "$DATA" "$OUT"

if [ ! -d "$DATA/endoscapes" ]; then
  wget -q -O "$DATA/endoscapes.zip" https://s3.unistra.fr/camma_public/datasets/endoscapes/endoscapes.zip
  unzip -q "$DATA/endoscapes.zip" -d "$DATA" && rm "$DATA/endoscapes.zip"
fi
# CholecSeg8k: attached notebook input (commit runs cannot attach datasets via kagglehub)
if [ -z "${CHOLECSEG:-}" ]; then
  HIT=$(find /kaggle/input -maxdepth 6 -type f -name "*_endo_color_mask.png" -print -quit 2>/dev/null || true)
  [ -n "$HIT" ] && CHOLECSEG=$(dirname "$(dirname "$(dirname "$HIT")")")
fi
if [ -z "${CHOLECSEG:-}" ]; then
  CHOLECSEG=$(python -c "import kagglehub; print(kagglehub.dataset_download('newslab/cholecseg8k'))" 2>/dev/null || true)
fi
if [ -n "${CHOLECSEG:-}" ]; then PRE=(--cholecseg "$CHOLECSEG"); echo "CholecSeg8k: $CHOLECSEG";
else PRE=(); echo "WARNING: CholecSeg8k not found, skipping pre-training"; fi

python scripts/data_stats.py --root "$DATA/endoscapes" --out "$OUT/data_stats.json"

# 1) segmentation — two CPU-fast candidates, keep the best on validation
for BB in efficientnet-b2 mit_b0; do
  python scripts/train_seg.py --endoscapes "$DATA/endoscapes" "${PRE[@]}" --backbone "$BB" \
    --epochs-pre 4 --epochs 60 --bs 16 --workers 4 --out "$OUT/seg_$BB.pt" | tail -n 3
done
BEST=$(python - <<PY
import torch
s = {b: torch.load(f"$OUT/seg_{b}.pt", weights_only=False)["val_dice_fg"] for b in ["efficientnet-b2", "mit_b0"]}
print(max(s, key=s.get)); print(s, file=__import__("sys").stderr)
PY
)
echo "best backbone: $BEST"
cp "$OUT/seg_$BEST.pt" "$OUT/seg.pt"

# 2) CVS heads — ablation (same frozen backbone, same seed)
for MODE in image nomask masks; do
  python scripts/train_cvs.py --endoscapes "$DATA/endoscapes" --seg-ckpt "$OUT/seg.pt" --mode "$MODE" \
    --epochs 12 --bs 32 --lr 1e-3 --workers 4 --out "$OUT/cvs_$MODE.pt" | tail -n 2
done

# 3) evaluation, threshold tuning (validation only), figures
python scripts/evaluate.py --endoscapes "$DATA/endoscapes" --ckpt-dir "$OUT" --metrics-out "$OUT/metrics.json" \
  --figures "$OUT/figures" --thresholds-out "$OUT/thresholds.yaml"

# 4) ONNX export
python scripts/export_onnx.py --ckpt "$OUT/cvs_masks.pt" --out "$OUT/choleguard.onnx"
echo "done"
