#!/bin/bash

if [ $# -lt 1 ]; then
    echo "usage: $0 [X220-compact|X220-enterprise]"
    exit 1
fi

if [ "$1" != "X220-compact" ] && [ "$1" != "X220-enterprise" ]
then
    echo "argument : [$1] must be [X220-compact] or [X220-enterprise]"
    exit 1
fi

CURRENT_PATH=$(pwd)
echo "CURRENT_PATH:$CURRENT_PATH , basename:$0 , system_name:$1"

system_name=$1

# OUTPUT=../../../../../SAPEON_SUBMISSION_$system_name

if [ -e "$OUTPUT" ]; then
    echo "$OUTPUT file or dir already exists!! move it"
    exit 1
fi

./run_inferencer.sh $system_name
./run_compliance.sh $system_name
