set -x

WORKDIR=`pwd`
REPODIR='<path/to/this/repo>'

PATTERN='[-a-zA-Z0-9_]*='
if [ $# -lt "0" ] ; then
    echo 'ERROR:'
    printf 'Please use following parameters:
    --code=<mlperf workload repo directory> 
    '
    exit 1
fi

if [ -z $DOWNLOAD_DATA_DIR ];then
    echo 'ERROR:'
    printf 'Please export DOWNLOAD_DATA_DIR=</path/of/kits19/data>'
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


proxy=$1

echo "Install dependencies"
echo "GCC minimum version: 11.1"
conda install -c conda-forge cmake jemalloc --yes
conda install intel-openmp mkl mkl-include mkl-service mkl_fft mkl_random  --no-update-deps --yes
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

if [ -d $code ];then
   REPODIR=$code
fi


echo "Install loadgen"
git clone https://github.com/mlcommons/inference.git
cd inference && git checkout r2.1
git log -1
git submodule update --init --recursive
cd loadgen
CFLAGS="-std=c++14" python setup.py install
cd ..


echo "Clone source code and Install"
echo "Install Intel Extension for PyTorch"
cd ${WORKDIR}
# clone Intel Extension for PyTorch
git clone https://github.com/intel/intel-extension-for-pytorch.git
cd intel-extension-for-pytorch
git checkout 1.9.0-rc
git submodule sync && git submodule update --init --recursive
cd third_party/mkl-dnn/
git checkout 840a6f14da0eb33fb38e7b43eccd0ac38b25c0ed
cd ../../
git apply ${REPODIR}/closed/Intel/code/3d-unet-99.9/pytorch-cpu-kits19/unet3d.diff
python setup.py install
cd ..

cd ${REPODIR}/closed/Intel/code/3d-unet-99.9/pytorch-cpu-kits19

pip install nibabel scipy pandas
make setup
make duplicate_kits19_case_00185

make preprocess_data
make preprocess_calibration_data

export LD_PRELOAD=${CONDA_PREFIX}/lib/libiomp5.so:$LD_PRELOAD
python trace_model.py
# python calibrate.py

