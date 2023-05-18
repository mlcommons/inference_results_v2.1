set -x
PATTERN='[-a-zA-Z0-9_]*='
if [ $# -lt "1" ] ; then
    echo 'ERROR:'
    printf 'Please use following parameters:
    --mode=<offline or server>
    --type=<performance or accuracy>
    --dtype=<int8, bf16 or fp32>
    '
    exit 1
fi


for i in "$@"
do
    case $i in
        --mode=*)
            mode=`echo $i | sed "s/${PATTERN}//"`;;
        --type=*)
            run_type=`echo $i | sed "s/${PATTERN}//"`;;
        --dtype=*)
            dtype=`echo $i | sed "s/${PATTERN}//"`;;
        *)
            echo "Parameter $i not recognized."; exit 1;;
    esac
done

WORK_DIR=$PWD
pwd
ls
source /workspace/setup_dataset.sh

if [ ${mode} = "offline" ]; then
    scenario="offline"
    source /workspace/setup_env_offline_AWS.sh
fi

if [ ${mode} = "server" ]; then
    scenario="server"
    source /workspace/setup_env_server_AWS.sh
fi

if [ ${run_type} = "perf" ];then
    accuracy=""
fi

if [ ${run_type} = "acc" ];then
    accuracy="accuracy"
fi

if [ ${dtype} = "fp32" ];then
    dtype=""
fi

echo ${mode} $BATCH_SIZE ${run_type} ${dtype} "mode"
sudo /workspace/run_clean.sh
/workspace/run_main.sh ${scenario} ${accuracy} ${dtype}
