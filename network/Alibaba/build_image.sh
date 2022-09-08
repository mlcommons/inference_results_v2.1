#!/bin/bash -xe
make build_docker DOCKER_TAG=nv CONTAINER_VOL=/demo NO_BUILD=1

img="vodla_mlperf:latest"

DOCKER_BUILDKIT=1 docker build -t $img $@ -f docker/Dockerfile.vodla .

