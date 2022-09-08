# Setup
Set up your system as detailed [here](https://github.com/krai/ck-qaic/blob/main/script/setup.docker/README.md).

# Benchmarking
```
SDK_VER=v1.7.1.12 POWER=yes SUT=r282_z93_q8e DOCKER=yes SERVER_ONLY=yes WORKLOADS="bert" $(ck find ck-qaic:script:run)/run_datacenter.sh
```
