if [[ -z "$1"  ||  -z "$2" ||  -z "$3"  ]] ; then
    echo "Usage:  $0    <image name or docker id> <Bash commands> <workdir> <optional: docker instance name> <optional: docker log file>"
    echo "Missing Docker image id or Bash commands.  exiting"
    exit -1
fi
if [ -z "$4" ] ; then
name="aikit_container"
else
name="$4"
fi

image_id="$1"
commands="$2"
workdir="$3"
#workdir="/opt/workdir/code/3d-unet-99.9/pytorch-cpu-kits19"
gpu_arg=""
GPU_DEV=/dev/dri
if [ -d "$GPU_DEV" ]; then
    echo "$GPU_DEV exists."
    gpu_arg=" --device=/dev/dri --ipc=host "
fi

## remove any previously running containers
docker rm -f "$name"

# mount the current directory at /work
this="${BASH_SOURCE-$0}"
mydir=$(cd -P -- "$(dirname -- "$this")" && pwd -P)

export DOCKER_RUN_ENVS="-e ftp_proxy=${ftp_proxy} -e FTP_PROXY=${FTP_PROXY} -e http_proxy=${http_proxy} -e HTTP_PROXY=${HTTP_PROXY} -e https_proxy=${https_proxy} -e HTTPS_PROXY=${HTTPS_PROXY} -e no_proxy=${no_proxy} -e NO_PROXY=${NO_PROXY} -e socks_proxy=${socks_proxy} -e SOCKS_PROXY=${SOCKS_PROXY}"

if [ -z "$5" ] ; then
docker run -a stdout  $DOCKER_RUN_ENVS  \
    --workdir "$workdir" \
    --volume $(pwd):/workspace \
    --privileged --init -it \
    --net host \
    --name "$name" $gpu_arg \
    --ipc host \
    "$image_id" \
    "$commands" --dataset-path /workspace/data/openimages
else
docker run  $DOCKER_RUN_ENVS  \
    --workdir "$workdir" \
    --volume $(pwd):/workspace \
    --privileged --init -it \
    --net host \
    --name "$name" $gpu_arg \
    --ipc host \
    "$image_id" \
    "$commands" --dataset-path /workspace/data/openimages
docker logs "$name" > "$5"
fi
