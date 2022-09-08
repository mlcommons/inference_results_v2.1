#!/usr/bin/env bash
CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"

CONDA_ENV_NAME=rn50-calenv-mlperf
export WORKDIR=${CUR_DIR}/${CONDA_ENV_NAME}
if [ -d ${WORKDIR} ]; then
	sudo rm -r ${WORKDIR}
fi

echo "Working directory is ${WORKDIR}"
mkdir -p ${WORKDIR}
cd ${WORKDIR}

conda create -n ${CONDA_ENV_NAME} python=3.9 --yes

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV_NAME}

echo "Installiing dependencies for RN50"
python -m pip install sklearn onnx
python -m pip install dataclasses
python -m pip install opencv-python
python -m pip install absl-py
python -m pip install matplotlib Pillow pycocotools

conda install typing_extensions --yes
conda config --add channels intel
conda install ninja pyyaml setuptools cmake cffi typing intel-openmp --yes
conda install -c intel mkl=2022.0.1 --yes
conda install -c intel mkl-include=2022.0.1 --yes
conda install -c conda-forge llvm-openmp --yes
conda install -c conda-forge jemalloc --yes
conda install future six requests dataclasses psutil --yes


export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

git clone https://github.com/pytorch/pytorch.git pytorch
cd pytorch
git fetch origin pull/76869/head:opt-cat
git checkout v1.12.0-rc7
git merge opt-cat --no-edit
git submodule sync
git submodule update --init --recursive
python -m pip install -r requirements.txt
python setup.py install

cd ${WORKDIR}

git clone https://github.com/intel/intel-extension-for-pytorch ipex-cpu-dev
cd ipex-cpu-dev
git checkout v1.12.0

git submodule sync
git submodule update --init --recursive


cp ${CUR_DIR}/input_output_aligned_scales.patch .
#cp ${CUR_DIR}/runtime_ignore_dequant_check.patch .

git apply input_output_aligned_scales.patch
#git apply runtime_ignore_dequant_check.patch
python -m pip install -r requirements.txt

python setup.py install

cd ${WORKDIR}

# Build torchvision
echo "Installiing torch vision"
git clone https://github.com/pytorch/vision
cd vision
python setup.py install

cd ${WORKDIR}

