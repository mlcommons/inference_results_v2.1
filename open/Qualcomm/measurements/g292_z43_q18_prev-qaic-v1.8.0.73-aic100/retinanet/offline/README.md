# Setup
Set up your system as detailed [here](https://github.com/krai/ck-qaic/blob/main/script/setup.aedk/README.md).

# Benchmarking
```
SDK_VER=v1.8.0.73 POWER=no SUT=g292_z43_q18_prev DOCKER=no OFFLINE_ONLY=yes WORKLOADS="retinanet" $(ck find ck-qaic:script:run)/run_edge.sh
```
