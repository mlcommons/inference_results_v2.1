#!/bin/bash

cd /work
source env.sh
make build_loadgen

BASH_NAME=$1
RUN_ARGS=$2

exec_shell="${BASH_NAME} ${RUN_ARGS}"
bash $exec_shell
