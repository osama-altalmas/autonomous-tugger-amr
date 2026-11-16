#!/usr/bin/env bash
# pack.sh [DATASET_DIR] : zip a generated dataset for upload to Colab
DIR=${1:-ai_data/markers_v1}
cd "$(dirname "$DIR")" && rm -f "$(basename "$DIR").zip" && zip -q -r "$(basename "$DIR").zip" "$(basename "$DIR")" -x '*/manifest.json' && ls -lh "$(basename "$DIR").zip"
