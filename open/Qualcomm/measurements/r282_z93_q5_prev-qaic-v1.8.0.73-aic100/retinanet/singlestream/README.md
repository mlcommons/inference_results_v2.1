# Setup
Set up your system as detailed [here](https://github.com/krai/ck-qaic/blob/main/script/setup.aedk/README.md).

# Benchmarking
```
SDK_VER=v1.8.0.73 POWER=no SUT=r282_z93_q5_prev DOCKER=no SINGLESTREAM_ONLY=yes WORKLOADS="retinanet" $(ck find ck-qaic:script:run)/run_edge.sh
```
