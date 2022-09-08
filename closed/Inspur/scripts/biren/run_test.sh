#!/bin/bash

cd /work
source env.sh
make biren

RUN_ARGS=$*
bash ${RUN_ARGS}
