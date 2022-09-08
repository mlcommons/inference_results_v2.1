# Setup
Set up your system as detailed [here](https://github.com/krai/ck-qaic/blob/main/script/setup.docker/README.md).

# Benchmarking
```
SDK_VER=v1.7.1.12 POWER=no SUT=r282_z93_q5 DOCKER=yes OFFLINE_ONLY=yes WORKLOADS="bert" $(ck find ck-qaic:script:run)/run_edge.sh
```
