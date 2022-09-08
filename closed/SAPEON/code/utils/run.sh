#!/bin/bash
RESULT_DIR=/home/shared/weight_result

if [ -e "$RESULT_DIR" ]; then
    echo "$RESULT_DIR file or dir already exists!! remove it!"
    exit 1
fi

# download calibration image data
wget https://cloud.sapeon.net:5043/mlperf/resnet50/recent_data/calib_img.tar
tar -xvf calib_img.tar
rm calib_img.tar 

# download resnet50_v1.onnx parameter
wget https://zenodo.org/record/2592612/files/resnet50_v1.onnx

# download docker image or use already exists
# if you want to download docker image, uncomment below 2 lines
wget https://cloud.sapeon.net:5043/mlperf/resnet50/recent_data/mlperf_quantization.tar
docker load -i mlperf_quantization.tar
rm mlperf_quantization.tar

# run quantization
mkdir result
docker run -it --rm -v $(pwd):/home/mlperf_script/ -w /home/mlperf_script/ mlperf_quantization ./run_process.sh

# copy result to shared dir /home/shared/
cp -r result $RESULT_DIR