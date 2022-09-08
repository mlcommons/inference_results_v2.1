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

ARG BASE_IMAGE=rockylinux:8.6
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
    gcc-toolset-11-gcc \
    gcc-toolset-11-gcc-c++ \
    && rm -rf /var/lib/yum/lists
RUN echo "alias ll='ls -l'" >> /root/.bashrc
RUN echo "source /opt/rh/gcc-toolset-11/enable" >> /root/.bashrc
ENV PATH /opt/conda/bin:$PATH

FROM dev-base as conda
ARG PYTHON_VERSION=3.9
RUN curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} && \
    /opt/conda/bin/conda install -c conda-forge -y cmake libunwind gperftools && \
    /opt/conda/bin/conda install -y intel-openmp mkl mkl-include numpy --no-update-deps && \
    /opt/conda/bin/conda clean -ya

FROM dev-base AS build
COPY --from=conda /opt/conda /opt/conda
ARG IPEX_VERSION=1.9.0-rc
ARG ONEDNN_VERSION=b5e06126da38bd8bee609d2965d62d30a53fe6b9
#COPY ./intel_inference_datacenter_v2-1/closed/Intel/code/dlrm-99.9/pytorch-cpu/dlrm.diff dlrm.diff
COPY ./code/dlrm-99.9/pytorch-cpu/dlrm.diff dlrm.diff
RUN --mount=type=cache,target=/opt/ccache \
    source /opt/rh/gcc-toolset-11/enable && \
    pip install absl-py tqdm numpy && \
    git clone -b ${IPEX_VERSION} https://github.com/intel/intel-extension-for-pytorch.git && \
    cd intel-extension-for-pytorch && \
    git submodule sync && git submodule update --init --recursive && \
    cd third_party/mkl-dnn/ && git checkout ${ONEDNN_VERSION} && cd ../.. && \
    git apply ../../../dlrm.diff && \
    python setup.py install
#    cd .. && python setup.py install

FROM dev-base as mp
COPY --from=build /opt/conda /opt/conda
WORKDIR /opt/workdir
#COPY ./intel_inference_datacenter_v2-1 intel_inference_datacenter_v2-1
COPY ./code/dlrm-99.9 code/dlrm-99.9
COPY ./calibration/dlrm-99.9 calibration/dlrm-99.9
RUN source /opt/rh/gcc-toolset-11/enable && \
    pip install -e git+https://github.com/mlperf/logging@1.0.0-rc4#egg=mlperf-logging && \
    git clone --recurse-submodules https://github.com/mlcommons/inference.git inference && \
    cd inference && git checkout r2.1 && \
    git submodule update --init --recursive && cd loadgen && \
    CFLAGS="-std=c++14" python setup.py install && cd ../..
#    pip install absl-py tqdm numpy

# ENV MALLOC_CONF "oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
ENV LD_PRELOAD "/opt/conda/lib/libiomp5.so"
ENV CONDA_PREFIX "/opt/conda"
