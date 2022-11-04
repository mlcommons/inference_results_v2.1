bash /workspace/make_preprocess.sh

number_threads=`nproc --all`
echo $number_threads
number_instance=$((number_threads/8))
echo $number_instance

if [ -z "${RUN_TYPE}" ]; then
    echo "NO RUN_TYPE ( perf/acc)found export RUN_TYPE=perf"
    export RUN_TYPE=perf
fi
echo $RUN_TYPE
bash run_mlperf.sh --type=${RUN_TYPE} \
	           --precision=int8 \
		   --user-conf=/workspace/user.conf \
		   --num-instance=$number_instance \
		   --cpus-per-instance=4 \
                   --scenario=Offline
