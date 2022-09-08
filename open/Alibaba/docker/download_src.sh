set -e
set -x

SRC_ROOT=$1
PATCH_PATH=./armnn.diff
mkdir -p $SRC_ROOT
cd $SRC_ROOT
git clone -b v3.12.0 https://github.com/google/protobuf.git protobuf
cd protobuf
git submodule update --init --recursive

cd $SRC_ROOT
git clone https://github.com/ARM-software/ComputeLibrary.git
cd ComputeLibrary/
git checkout v22.05

cd $SRC_ROOT
wget -O flatbuffers-1.12.0.tar.gz https://github.com/google/flatbuffers/archive/v1.12.0.tar.gz
tar xf flatbuffers-1.12.0.tar.gz

cd $SRC_ROOT
git clone https://github.com/onnx/onnx.git
cd onnx
git fetch https://github.com/onnx/onnx.git 553df22c67bee5f0fe6599cff60f1afc6748c635 && git checkout FETCH_HEAD

cd $SRC_ROOT
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow/
git checkout tags/v2.5.1

cd $SRC_ROOT
git clone https://github.com/ARM-software/armnn.git
cd armnn
git checkout branches/armnn_22_05
patch -p1 <$PATCH_PATH
