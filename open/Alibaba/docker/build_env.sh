set -e
set -x
SRC_ROOT=$1
BUILD_ROOT=$2
mkdir -p $BUILD_ROOT && cd $BUILD_ROOT

cp -r $SRC_ROOT/* ./

cd $BUILD_ROOT/protobuf
./autogen.sh
mkdir arm64_build
cd arm64_build
../configure --host=aarch64-linux \
  --prefix=$BUILD_ROOT/google/arm64_pb_install \
  --with-protoc=$BUILD_ROOT/google/arm64_pb_install/bin/protoc
make install -j48
cd ..

cd $BUILD_ROOT/ComputeLibrary
scons -j128 arch=armv8.2-a build=native opencl=false neon=true openmp=false cppthreads=False \
  extra_cxx_flags="-fPIC" \
  build_dir=$BUILD_ROOT/ComputeLibrary/build \
  install_dir=$BUILD_ROOT/ComputeLibrary/install \
  Werror=0

cd $BUILD_ROOT
cd flatbuffers-1.12.0
rm -f CMakeCache.txt
mkdir build
cd build
CXXFLAGS="-fPIC" cmake .. -DFLATBUFFERS_BUILD_FLATC=1 \
  -DCMAKE_INSTALL_PREFIX:PATH=$BUILD_ROOT/flatbuffers \
  -DFLATBUFFERS_BUILD_TESTS=0
make all install -j48

cd $BUILD_ROOT
cd onnx
LD_LIBRARY_PATH=$BUILD_ROOT/google/arm64_pb_install/lib:$LD_LIBRARY_PATH \
  $BUILD_ROOT/google/arm64_pb_install/bin/protoc \
  onnx/onnx.proto --proto_path=. --proto_path=../google/arm64_pb_install/include --cpp_out $BUILD_ROOT/onnx

cd $BUILD_ROOT
mkdir tflite
cd tflite
cp ../tensorflow/tensorflow/lite/schema/schema.fbs .
../flatbuffers-1.12.0/build/flatc -c --gen-object-api --reflect-types --reflect-names schema.fbs

cd $BUILD_ROOT
rm -rf build_armnn
mkdir build_armnn
cd build_armnn
cmake ../armnn/. \
  -DCMAKE_INSTALL_PREFIX:PATH=$BUILD_ROOT/install \
  -DARMCOMPUTE_ROOT=$BUILD_ROOT/ComputeLibrary \
  -DARMCOMPUTE_BUILD_DIR=$BUILD_ROOT/ComputeLibrary/build/ \
  -DARMCOMPUTENEON=1 -DARMCOMPUTECL=0 -DARMNNREF=1 \
  -DONNX_GENERATED_SOURCES=$BUILD_ROOT/onnx \
  -DBUILD_ONNX_PARSER=1 \
  -DBUILD_TF_LITE_PARSER=1 \
  -DTF_LITE_GENERATED_PATH=$BUILD_ROOT/tflite \
  -DFLATBUFFERS_ROOT=$BUILD_ROOT/flatbuffers \
  -DFLATC_DIR=$BUILD_ROOT/flatbuffers-1.12.0/build \
  -DPROTOBUF_ROOT=$BUILD_ROOT/google/arm64_pb_install \
  -DPROTOBUF_ROOT=$BUILD_ROOT/google/arm64_pb_install/ \
  -DPROTOBUF_LIBRARY_DEBUG=$BUILD_ROOT/google/arm64_pb_install/lib/libprotobuf.so.23.0.0 \
  -DPROTOBUF_LIBRARY_RELEASE=$BUILD_ROOT/google/arm64_pb_install/lib/libprotobuf.so.23.0.0 \
  -DBUILD_PYTHON_SRC=1 \
  -DBUILD_PYTHON_WHL=1 \
  -DBUILD_ARMNN_DESERIALIZER=1 \
  -DBUILD_ARMNN_SERIALIZER=1

make -j
make install
