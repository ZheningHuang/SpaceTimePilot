#!/bin/bash
# Run all 5 time-mode inferences for moved_cam2moved_cam_extended (32 scenes).
# Runs sequentially; stops immediately on any failure.

set -e

CKPT="checkpoints/SpacetimePilot_1.3B_v1.ckpt"
CFG_DIR="config/evaluation"
OUT_BASE="./results/moved_cam2moved_cam_extended"

# Ordered: bullet-time, forward, reverse, slow-motion, zigzag
declare -a MODES=(
    "fixed_10"   # bullet time (freeze at frame 40)
    "normal"     # forward playback (0→80)
    "reverse"    # reverse playback (80→0)
    "slowmo"     # slow motion 0.5× (0→40 stretched over 81 frames)
    "zigzag"     # zigzag (0→40→0)
)

declare -A TIME_MODE=(
    ["fixed_10"]="fixed_10"
    ["normal"]="normal"
    ["reverse"]="reverse"
    ["slowmo"]="repeat_0to40_double"
    ["zigzag"]="zigzag_0_10_0"
)

TOTAL=${#MODES[@]}
IDX=0

for MODE in "${MODES[@]}"; do
    IDX=$((IDX + 1))
    echo ""
    echo "========================================================"
    echo "  [$IDX/$TOTAL]  mode: $MODE  (time_mode: ${TIME_MODE[$MODE]})"
    echo "========================================================"
    python inference_batch.py \
        --config "$CFG_DIR/${MODE}.yaml" \
        -ckpt   "$CKPT" \
        --output_dir "$OUT_BASE/$MODE"
done

echo ""
echo "========================================================"
echo "  All $TOTAL inferences complete."
echo "  Results saved under $OUT_BASE/"
echo "========================================================"
