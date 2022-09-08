FROM ubuntu:21.10
RUN apt update 
RUN apt install -y software-properties-common gcc gdb linux-tools-generic libtinfo5 llvm-13 clang-13 libclang-13-dev
RUN apt install -y  python3-pip python3-dev python3-setuptools libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev automake  
RUN ln -s /usr/bin/python3.9 /usr/bin/python && rm -f /usr/bin/pip && ln -s /usr/bin/pip3 /usr/bin/pip
RUN apt install -y libopenblas-dev libtool swig scons git vim wget ninja-build gfortran numactl curl unzip
RUN pip install torch==1.10.0 torchvision==0.10.0 -f https://torch.kmtea.eu/whl/stable-cn.html
RUN pip install pyyaml pytest pybind11 cython pythran tornado psutil cloudpickle decorator attrs synr==0.6.0 tflite==2.4.0 onnx==1.10.2 xgboost==1.5.0
RUN apt-get update && apt-get install -y  python3-opencv