#!/usr/bin/env bash
# Batch-run run_video_inference.py on every .mp4 under INPUT_DIR.
# Skips any scene whose output dir already has a complete run_config.json.

set -euo pipefail

INPUT_DIR="${INPUT_DIR:-/home/builder/workspace/video_eval/manual/sf_17/20260402_sf_exp_17_ema_checkpoint-20500_20260414_155216}"
OUT_ROOT="${OUT_ROOT:-/home/builder/workspace/lingbot-map/outputs/sf17}"
MODEL="${MODEL:-/home/builder/workspace/lingbot-map/checkpoints/lingbot-map-long.pt}"
FPS="${FPS:-16}"
GPU="${GPU:-7}"

cd /home/builder/workspace/lingbot-map
# shellcheck disable=SC1091
source .venv/bin/activate

mkdir -p "$OUT_ROOT"
BATCH_LOG="$OUT_ROOT/batch.log"
: > "$BATCH_LOG"

mapfile -t VIDEOS < <(find "$INPUT_DIR" -maxdepth 1 -type f -name '*.mp4' | sort)
echo "Found ${#VIDEOS[@]} videos" | tee -a "$BATCH_LOG"

t_total_start=$(date +%s)
idx=0
for video in "${VIDEOS[@]}"; do
    idx=$((idx + 1))
    name="$(basename "${video%.mp4}")"
    out_dir="$OUT_ROOT/$name"
    log="$out_dir/run.log"

    echo "" | tee -a "$BATCH_LOG"
    echo "[$idx/${#VIDEOS[@]}] $name" | tee -a "$BATCH_LOG"

    if [[ -f "$out_dir/run_config.json" && -f "$out_dir/input_and_map.mp4" ]]; then
        echo "  already complete; skipping." | tee -a "$BATCH_LOG"
        continue
    fi

    mkdir -p "$out_dir"
    t_start=$(date +%s)
    if CUDA_VISIBLE_DEVICES="$GPU" python run_video_inference.py \
        --video_path "$video" \
        --model_path "$MODEL" \
        --output_dir "$out_dir" \
        --fps "$FPS" \
        --skip_per_frame_npz \
        > "$log" 2>&1; then
        t_end=$(date +%s)
        secs=$((t_end - t_start))
        printf "  OK in %ds  ->  %s\n" "$secs" "$out_dir" | tee -a "$BATCH_LOG"
    else
        t_end=$(date +%s)
        secs=$((t_end - t_start))
        echo "  FAILED after ${secs}s (see $log)" | tee -a "$BATCH_LOG"
        tail -5 "$log" | sed 's/^/    /' | tee -a "$BATCH_LOG" || true
    fi
done

t_total_end=$(date +%s)
total=$((t_total_end - t_total_start))
echo "" | tee -a "$BATCH_LOG"
printf "Total elapsed: %dm%ds\n" $((total / 60)) $((total % 60)) | tee -a "$BATCH_LOG"
