#!/bin/env bash

export DOCKER_BUILD_ARGS="--build-arg ftp_proxy=${ftp_proxy} --build-arg FTP_PROXY=${FTP_PROXY} --build-arg http_proxy=${http_proxy} --build-arg HTTP_PROXY=${HTTP_PROXY} --build-arg https_proxy=${https_proxy} --build-arg HTTPS_PROXY=${HTTPS_PROXY} --build-arg no_proxy=${no_proxy} --build-arg NO_PROXY=${NO_PROXY} --build-arg socks_proxy=${socks_proxy} --build-arg SOCKS_PROXY=${SOCKS_PROXY}"

export DOCKER_RUN_ENVS="--env ftp_proxy=${ftp_proxy} --env FTP_PROXY=${FTP_PROXY} --env http_proxy=${http_proxy} --env HTTP_PROXY=${HTTP_PROXY} --env https_proxy=${https_proxy} --env HTTPS_PROXY=${HTTPS_PROXY} --env no_proxy=${no_proxy} --env NO_PROXY=${NO_PROXY} --env socks_proxy=${socks_proxy} --env SOCKS_PROXY=${SOCKS_PROXY}"

function usage() {
  echo "To build any of the workflow containers, try:"
  echo "$ bash build_workflow_containers.sh <WORKFLOW_NAME>"
  echo "Supported workflows are: BERT-99, DLRM-99.9, RESNET50, RNNT and SSD-RESNET34"
  echo "To build all workflow containers try:"
  echo "$ bash build_workflow_containers.sh --all"
}

function build_workflow_container() {
  local WORKFLOW=$1
  pushd ${WORKFLOW}/pytorch-cpu*/docker
  bash build_${WORKFLOW}_container.sh
  popd
}

WORKFLOW=$(echo $1 | tr '[:upper:]' '[:lower:]')
case ${WORKFLOW} in
  bert-99|dlrm-99.9|resnet50|retinanet|3d-unet-99.9)
    build_workflow_container ${WORKFLOW}
    ;;
  --all)
    WORKFLOWS=(bert-99 dlrm-99.9 resnet50 3d-unet-99.9 retinanet)
    for WORKFLOW in ${WORKFLOWS[@]}; do
      build_workflow_container ${WORKFLOW}
    done
    ;;
 *)
   usage;
   ;;
esac
