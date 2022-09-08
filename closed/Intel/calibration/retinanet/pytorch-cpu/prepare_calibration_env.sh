#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"

CONDA_ENV_NAME=retinanet-calibration-env
export WORKDIR=${DIR}/${CONDA_ENV_NAME}
if [ -d ${WORKDIR} ]; then
	sudo rm -r ${WORKDIR}
fi

echo "Working directory is ${WORKDIR}"
mkdir -p ${WORKDIR}
cd ${WORKDIR}

#exit 1

PATTERN='[-a-zA-Z0-9_]*='
if [ $# -lt "0" ] ; then
    echo 'ERROR:'
    printf 'Please use following parameters:
    --code=<mlperf workload repo directory> 
    '
    exit 1
fi

for i in "$@"
do
    case $i in
        --code=*)
            code=`echo $i | sed "s/${PATTERN}//"`;;
        *)
            echo "Parameter $i not recognized."; exit 1;;
    esac
done

if [ -d $code ];then
    REPODIR=$code
fi

source ~/anaconda3/etc/profile.d/conda.sh
conda create -n ${CONDA_ENV_NAME} python=3.9 --yes
conda activate ${CONDA_ENV_NAME}

echo "Installiing dependencies for Retinanet"
python -m pip install Pillow pycocotools==2.0.2
python -m pip install opencv-python
python -m pip install absl-py
python -m pip install fiftyone

#conda install typing_extensions --yes
conda config --add channels intel
conda install setuptools cmake intel-openmp --yes
conda install -c intel mkl=2022.0.1 --yes
conda install -c intel mkl-include=2022.0.1 --yes
conda install -c conda-forge llvm-openmp --yes
conda install -c conda-forge jemalloc --yes

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

#build pytorch and intel-pytorch-extension
git clone https://github.com/pytorch/pytorch.git pytorch
cd pytorch

git checkout v1.12.0-rc7

git submodule sync
git submodule update --init --recursive
python setup.py install

cd ${WORKDIR}
git clone https://github.com/intel/intel-extension-for-pytorch.git ipex-cpu-dev
cd ipex-cpu-dev

git checkout mlperf/retinanet

git submodule sync
git submodule update --init --recursive

python setup.py install
export IPEX_PATH=${PWD}/build/Release/packages/intel_extension_for_pytorch

export TORCH_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'`

cd ${WORKDIR}

# Build torchvision
echo "Installiing torch vision"
git clone https://github.com/pytorch/vision
cd vision
python setup.py install


