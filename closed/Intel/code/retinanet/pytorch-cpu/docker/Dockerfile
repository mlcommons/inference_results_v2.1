# syntax = docker/dockerfile:experimental
# based onhttps://github.com/pytorch/pytorch/blob/master/Dockerfile
# 
# NOTE: To build this you will need a docker version > 18.06 with
#       experimental enabled and DOCKER_BUILDKIT=1
#
#       If you do not use buildkit you are not going to have a good time
#
#       For reference: 
#           https://docs.docker.com/develop/develop-images/build_enhancements/

ARG BASE_IMAGE=rockylinux:8.5
FROM ${BASE_IMAGE} AS dev-base
RUN --mount=type=cache,id=yum-dev,target=/var/cache/yum \
    DEBIAN_FRONTEND=noninteractive dnf install -y \
    ca-certificates \
    git \
    curl \
    vim \
    numactl \
    cmake \
    sudo \
    wget \
    mesa-libGL \
    gcc-toolset-11-gcc \
    gcc-toolset-11-gcc-c++ \
    && rm -rf /var/lib/yum/lists/*
RUN echo "alias ll='ls -l'" >> /root/.bashrc
RUN echo "source /opt/rh/gcc-toolset-11/enable" >> /root/.bashrc
RUN mkdir /opt/ccache
ENV PATH /opt/conda/bin:$PATH

FROM dev-base as conda
ARG PYTHON_VERSION=3.9
RUN curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda config --add channels intel && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} && \
    /opt/conda/bin/conda install -y setuptools cmake intel-openmp && \
    /opt/conda/bin/conda install -c intel mkl=2022.0.1 mkl-include=2022.0.1 -y && \
    /opt/conda/bin/conda install -c conda-forge llvm-openmp jemalloc -y && \
    /opt/conda/bin/conda clean -ya

FROM dev-base AS build
COPY --from=conda /opt/conda /opt/conda
ARG PYTORCH_VERSION=v1.12.0-rc7
RUN --mount=type=cache,target=/opt/ccache \
    source /opt/rh/gcc-toolset-11/enable && \
    pip install Pillow pycocotools==2.0.2 opencv-python absl-py fiftyone && \
    git clone https://github.com/pytorch/pytorch.git pytorch && cd pytorch && \
    git checkout ${PYTORCH_VERSION} && git submodule sync && git submodule update --init --recursive && \
    python setup.py install && cd ..

FROM dev-base as mp-base
COPY --from=build /opt/conda /opt/conda
ENV CONDA_PREFIX "/opt/conda"
WORKDIR /opt/workdir
COPY ./code/retinanet code/retinanet
COPY ./calibration/retinanet calibration/retinanet
ARG IPEX_VERSION=mlperf/retinanet
ARG INFERENCE=5cc4995215bba2d019e62ded3639d7beacc9cff3
RUN source /opt/rh/gcc-toolset-11/enable && \
    git clone https://github.com/pytorch/vision && \
    cd vision && python setup.py install && cd .. && \
    git clone https://github.com/intel/intel-extension-for-pytorch.git ipex-cpu-dev && \
    cd ipex-cpu-dev && git checkout ${IPEX_VERSION} && \
    git submodule sync && git submodule update --init --recursive && \
    python setup.py install && \
    export IPEX_PATH=${PWD}/build/Release/packages/intel_extension_for_pytorch && cd .. && \
    cd code/retinanet/pytorch-cpu/ && \
    git clone -b r2.1 https://github.com/mlcommons/inference.git mlperf_inference && \
    cd mlperf_inference && git checkout ${INFERENCE} && git submodule update --init --recursive && \
    cd loadgen && mkdir build && cd build && cmake .. && make && cd .. && \
    CFLAGS="-std=c++14" python setup.py install && export LOADGEN_DIR=${PWD} && cd ../.. && \
    git clone https://github.com/opencv/opencv.git && cd opencv && git checkout 4.x && \
    mkdir build && cd build && cmake .. && make -j$(nproc) && \
    export OPENCV_DIR=${PWD} && cd ../.. && \
    git clone https://github.com/Tencent/rapidjson.git && cd rapidjson && git checkout e4bde977 && \
    export RAPIDJSON_INCLUDE_DIR=${PWD}/include && cd .. && \
    git clone https://github.com/gflags/gflags.git && cd gflags && mkdir build && cd build && \
    cmake .. && make && export GFLAGS_DIR=${PWD} && cd ../.. && \
    export LD_LIBRARY_PATH=${OPENCV_DIR}/lib:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH} && \
    export LIBRARY_PATH=${CONDA_PREFIX}/lib:${LIBRARY_PATH} && \
    rm -rf /opt/conda/lib/cmake/mkl && \
    export TORCH_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` && \
    cmake -DCMAKE_PREFIX_PATH=${TORCH_PATH} \
        -DLOADGEN_DIR=${LOADGEN_DIR} \
        -DOpenCV_DIR=${OPENCV_DIR} \
        -DRapidJSON_INCLUDE_DIR=${RAPIDJSON_INCLUDE_DIR} \
        -Dgflags_DIR=${GFLAGS_DIR} \
        -DINTEL_EXTENSION_FOR_PYTORCH_PATH=${IPEX_PATH} \
        -B${PWD}/build \
        -H${PWD}/src && \
    cmake --build ${PWD}/build --config Release

ENV MALLOC_CONF "oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
ENV LD_PRELOAD "/opt/conda/lib/libjemalloc.so":"/opt/conda/lib/libiomp5.so":${LD_PRELOAD}
ENV ENV_DEPS_DIR "/opt/workdir/code/retinanet/pytorch-cpu"
ENV CONDA_PREFIX "/opt/conda"
