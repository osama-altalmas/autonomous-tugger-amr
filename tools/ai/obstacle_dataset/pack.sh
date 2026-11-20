#!/usr/bin/env bash
# Zip an obstacle dataset for upload to Colab/Kaggle (into ai_data/).
# Usage: pack.sh [v1|v2|v3|v4|v5]   (v1 also zips the v1 test set; v3 also its test set and LOCO;
#        v4 its test set, LOCO v4 and the other real photos: COCO people and the rf_* sets)
set -euo pipefail
v=${1:-v3}
cd "$(dirname "$0")/../../../ai_data"
rm -f "obstacles_$v.zip"
zip -q -r "obstacles_$v.zip" "obstacles_$v/images" "obstacles_$v/labels" "obstacles_$v/data.yaml"
if [ "$v" = v1 ]; then
  rm -f obstacles_v1_test.zip
  zip -q -r obstacles_v1_test.zip obstacles_v1_test/images obstacles_v1_test/labels
fi
if [ "$v" = v3 ]; then
  rm -f obstacles_v3_test.zip loco_yolo.zip
  zip -q -r obstacles_v3_test.zip obstacles_v3_test/images obstacles_v3_test/labels
  zip -q -r loco_yolo.zip loco_yolo/images loco_yolo/labels
fi
if [ "$v" = v4 ]; then
  rm -f obstacles_v4_test.zip loco_yolo_v4.zip real_extra_v4.zip
  zip -q -r obstacles_v4_test.zip obstacles_v4_test/images obstacles_v4_test/labels
  zip -q -r loco_yolo_v4.zip loco_yolo_v4/images loco_yolo_v4/labels
  zip -q -r real_extra_v4.zip coco_people/images coco_people/labels rf_*/images rf_*/labels
fi
if [ "$v" = v5 ]; then
  rm -f obstacles_v5_test.zip loco_yolo_v5.zip real_extra_v5.zip
  zip -q -r obstacles_v5_test.zip obstacles_v5_test/images obstacles_v5_test/labels
  zip -q -r loco_yolo_v5.zip loco_yolo_v5/images loco_yolo_v5/labels
  zip -q -r real_extra_v5.zip coco_people/images coco_people/labels rf_*/images rf_*/labels
fi
ls -la obstacles_*.zip real_extra_*.zip loco_yolo.zip 2>/dev/null
