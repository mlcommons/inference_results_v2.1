#!/bin/bash

DATA_PATH=data
if [ -e "$DATA_PATH/preprocessed_data" ]; then
    echo "$DATA_PATH/preprocessed_data file or dir already exists!! if you want to preprocess again, remove it!"
    exit 1
else
    ln -s /home/shared/ $DATA_PATH
fi

# we already prepare imangenet data at host system `/home/shared`.
# if you want download data, uncomment below block
#######
# mkdir $DATA_PATH
# cd $DATA_PATH
# wget https://cloud.sapeon.net:5043/mlperf/resnet50/recent_data/ILSVRC2012_img_val.tar
# mkdir ILSVRC2012_val
# tar -xvf ILSVRC2012_img_val.tar -C ILSVRC2012_val
# rm ILSVRC2012_img_val.tar
# wget https://cloud.sapeon.net:5043/mlperf/resnet50/recent_data/val.txt
# cd ../
#######

pip3 install tqdm numpy opencv-python
python3 preprocess_input.py -i $DATA_PATH/ILSVRC2012_val/ -o $DATA_PATH/preprocessed_data
