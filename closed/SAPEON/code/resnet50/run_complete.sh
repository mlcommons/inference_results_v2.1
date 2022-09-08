#!/bin/bash

CURRENT_PATH=$(pwd)
echo "CURRENT_PATH:$CURRENT_PATH , basename:$0 , OUTPUT:$1"

ROOT_DIR=../../../../..
MLPERF_DIR=./MlperfInference

OUTPUT=$1

echo "truncate_accuracy_log.py start"
python3 $MLPERF_DIR/tools/submission/truncate_accuracy_log.py --input ../../../../ --output $OUTPUT --submitter SAPEON
echo "truncate_accuracy_log.py end"

echo "submission-checker start"
python3 $MLPERF_DIR/tools/submission/submission-checker.py --input $OUTPUT > $OUTPUT/submission-checker.log 2>&1
echo "submission-checker end"