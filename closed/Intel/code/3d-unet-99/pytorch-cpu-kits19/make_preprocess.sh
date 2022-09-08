if [ -z $https_proxy ]; then
	echo "please export https_proxy first!"
	exit 1
fi

export DOWNLOAD_DATA_DIR=/data/mlperf_data/3dunet/kits19/data
make setup
make duplicate_kits19_case_00185
make preprocess_data
make preprocess_calibration_data
export LD_PRELOAD=${CONDA_PREFIX}/lib/libiomp5.so:$LD_PRELOAD
python trace_model.py
