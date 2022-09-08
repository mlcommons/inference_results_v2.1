#!/bin/bash

# global vars
REPO_AUTHOR=intel
REPO_NAME=neural-compressor
# REPO_NAME=nc
# COMMIT_ID=31d3d23bc17e5c4f4973979c9924b80d6be0c571
COMMIT_ID=0803
CALI_DATA=ILSVRC2012_img_calib.tar.gz
CURRENT_DIR=$(pwd)
UPPER_DIR=$(dirname $(pwd))
PATCH_DIR="$CURRENT_DIR/patches"
SRCFILE_DIR="$CURRENT_DIR/modified_code"

MODEL_DIR="$CURRENT_DIR/models"
MODEL_LINK=https://zenodo.org/record/4735647/files/resnet50_v1.onnx # opset=11
# MODEL_LINK=https://zenodo.org/record/2592612/files/resnet50_v1.onnx # opset=9

SRC_YAML=$CURRENT_DIR"/"modified_code/examples/onnxrt/image_recognition/resnet50/quantization/ptq/resnet50_v1_5_mlperf.yaml

REPO_PYTHONPATH=$CURRENT_DIR"/"$REPO_NAME
MLPerf_YAML=$REPO_PYTHONPATH"/"examples/onnxrt/image_recognition/resnet50/quantization/ptq/resnet50_v1_5_mlperf.yaml
DATASET_TAR=$CURRENT_DIR"/"ILSVRC2012_img_calib.tar.gz
DATASET=$CURRENT_DIR
IMAGE_LIST=$CURRENT_DIR"/"ILSVRC2012_img_val_labels.txt


# ## create a folder and save patches to this folder (TODO)
# # if patches do not exist, generate them
# if [ ! -d "$PATCH_DIR" ]; then
#   echo " $PATCH_DIR does not exist. Create a PATCH folder ... "
#   mkdir -p $PATCH_DIR

#   git clone "https://github.com/$REPO_AUTHOR/$REPO_NAME.git"
#   pushd $REPO_NAME 
#   # git submodule update --init --recursive 
#   git checkout $COMMIT_ID
  
#   cp -r $SRCFILE_DIR"/"* .
#   git add . && git commit -m "[Fix] generate patches"
#   # generate patches
#   git format-patch -M master
#   git format-patch -n $COMMIT_ID -o $PATCH_DIR

#   popd
#   if [ -d $REPO_NAME ]; then
#     rm -rf $REPO_NAME
#   fi
# fi

pip install -r requirements.txt -i https://pypi.douban.com/simple
## download model and do preprocessing
# download ONNX model in FP32  
if [ ! -f $MODEL_DIR"/"resnet50_v1.onnx ]; then
  mkdir -p $MODEL_DIR
  echo " ====== Download MLPerf inference resnet50-v1.5 ONNX model ====== "
  wget -c $MODEL_LINK -O $MODEL_DIR"/"resnet50_v1.onnx
  echo " ====== saved model to $MODEL_DIR  ====== "
fi
# preprocess model to add an equivalent conv
echo " ====== MLPerf inference resnet50-v1.5 ONNX model existing ====== "
python resnet50_preprocess_1.py $MODEL_DIR"/"resnet50_v1.onnx $MODEL_DIR"/"resnet50_mlperf_equal_conv.onnx
echo " ====== Saved preprocessed FP32 model to $MODEL_DIR/resnet50_mlperf_equal_conv.onnx ====== "


# ## setup environment (TODO) update neural-compressor commit id 
# echo " $PATCH_DIR exists. Try to find PATCH files and patch them ... "
# git clone "https://github.com/$REPO_AUTHOR/$REPO_NAME.git"
pushd $REPO_NAME
# # git submodule update --init --recursive 
git checkout $COMMIT_ID
# git am $PATCH_DIR"/"*.patch

# install requirements & neural-compressor and leave the directory
pip install PyYAML -i https://pypi.douban.com/simple > /dev/null
pip install -r requirements.txt -i https://pypi.douban.com/simple > /dev/null
# python setup.py install
export PYTHONPATH=$PYTHONPATH:$REPO_PYTHONPATH

popd

tar zxvf $DATASET_TAR > /dev/null # -C $DATASET 
# cp ILSVRC2012_img_val_labels.txt $IMAGE_LIST

# patch MLPerf yaml file
# cp $SRC_YAML $MLPerf_YAML
python patch_yaml.py --yaml_path $MLPerf_YAML --data_path $DATASET"/"ILSVRC2012_img_calib --image_list $IMAGE_LIST

# calibrate
python $REPO_PYTHONPATH"/"examples/onnxrt/image_recognition/resnet50/quantization/ptq/main.py \
    --model_path $MODEL_DIR"/"resnet50_mlperf_equal_conv.onnx \
    --config $MLPerf_YAML \
    --output_model $MODEL_DIR"/"resnet50_mlperf_equal_conv_output.onnx

# post-process
python resnet50_preprocess_2.py --input_model $MODEL_DIR"/"resnet50_mlperf_equal_conv_output.onnx \
    --output_model $MODEL_DIR"/"resnet50_int8_delete_quantize_linear_v2.onnx
