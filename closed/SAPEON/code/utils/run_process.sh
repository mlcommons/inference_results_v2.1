#!/bin/bash
export PYTHONPATH=.
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/lib
export LD_LIBRARY_PATH

INPUT_DATA=resnet50_v1.onnx
GRAPH=sapeon_graph
CALIB=calib_tbl

if [ -e "$GRAPH" ]; then
    echo "$GRAPH file or dir already exists!! remove it!"
    exit 1
fi
if [ -e "$CALIB" ]; then
    echo "$CALIB file or dir already exists!! remove it!"
    exit 1
fi

./sapeon_front -i $INPUT_DATA -o $GRAPH
chmod 777 $GRAPH
./sapeon_calibrator image_util $GRAPH $CALIB
chmod 777 $CALIB
./sapeon_compile $GRAPH $CALIB

echo "$0 complete"