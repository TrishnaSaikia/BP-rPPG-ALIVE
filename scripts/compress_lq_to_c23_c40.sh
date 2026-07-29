#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

in_dir=${1:-raw_videos/LQ}
out_c23=${2:-raw_videos/C_23}
out_c40=${3:-raw_videos/C_40}
mkdir -p "$out_c23" "$out_c40"

files=("$in_dir"/*.avi)
if (( ${#files[@]} == 0 )); then
  echo "No AVI videos found in $in_dir" >&2
  exit 1
fi

for file in "${files[@]}"; do
  base=$(basename "$file")
  ffmpeg -y -i "$file" -c:v libx264 -crf 23 -preset medium -an "$out_c23/$base"
  ffmpeg -y -i "$file" -c:v libx264 -crf 40 -preset medium -an "$out_c40/$base"
done
