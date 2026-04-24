#!/usr/bin/env bash
# Re-encode every outputs/**/*.mp4 to HEVC (libx265) in-place.
#
# Strategy:
#   1. Find every .mp4 under OUTPUT_ROOT.
#   2. For each, encode to a .tmp.x265.mp4 sibling.
#   3. If the new file is decodable AND strictly smaller, replace the original.
#   4. Otherwise, keep the original and delete the tmp file.
#
# Runs JOBS encoders at a time. libx265 itself is multi-threaded, so we keep
# concurrency modest.

set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/builder/workspace/lingbot-map/outputs}"
CRF="${CRF:-28}"               # 28 = good balance; lower = higher quality/size
PRESET="${PRESET:-medium}"     # slower presets compress better
JOBS="${JOBS:-4}"

encode_one() {
    local src="$1"
    local dst="${src%.mp4}.x265.tmp.mp4"
    local size_before size_after
    size_before=$(stat -c%s "$src")

    # -hide_banner -loglevel error: keep stderr quiet
    # -map_metadata 0: preserve rotation, etc.
    # -c:a copy (if present): passthrough audio (our mp4s are video-only, but harmless)
    if ! ffmpeg -hide_banner -loglevel error -y -i "$src" \
            -map 0:v:0 -c:v libx265 -crf "$CRF" -preset "$PRESET" \
            -pix_fmt yuv420p -tag:v hvc1 \
            -map_metadata 0 "$dst" 2> >(grep -v 'Application provided' >&2); then
        echo "ENCODE FAILED: $src"
        rm -f "$dst"
        return 1
    fi

    # Sanity: decodable
    if ! ffprobe -v error -select_streams v:0 -show_entries stream=codec_name \
            -of default=noprint_wrappers=1:nokey=1 "$dst" > /dev/null; then
        echo "DECODE CHECK FAILED: $src"
        rm -f "$dst"
        return 1
    fi

    size_after=$(stat -c%s "$dst")
    local mb_before mb_after pct
    mb_before=$(awk -v s="$size_before" 'BEGIN{printf "%.1f", s/1048576}')
    mb_after=$(awk -v s="$size_after"  'BEGIN{printf "%.1f", s/1048576}')
    if (( size_after < size_before )); then
        mv -f "$dst" "$src"
        pct=$(( 100 * size_after / size_before ))
        printf '  %s: %s MB -> %s MB (%d%%)\n' "$src" "$mb_before" "$mb_after" "$pct"
    else
        rm -f "$dst"
        printf '  %s: skipped (HEVC not smaller: %s MB -> %s MB)\n' \
            "$src" "$mb_before" "$mb_after"
    fi
}

export -f encode_one
export CRF PRESET

mapfile -t FILES < <(find "$OUTPUT_ROOT" -type f -name '*.mp4' | sort)
echo "Re-encoding ${#FILES[@]} files to HEVC (crf=$CRF preset=$PRESET, jobs=$JOBS)..."
printf '  %s\n' "${FILES[@]}"

# xargs -P N: run N encode_one invocations in parallel.
printf '%s\0' "${FILES[@]}" | xargs -0 -n1 -P"$JOBS" bash -c 'encode_one "$0"'

echo "Done."
