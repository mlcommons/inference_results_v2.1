# Setup
Set up your system as detailed [here](https://github.com/krai/ck-qaic/blob/main/script/setup.aedk/README.md).

# Benchmarking
```
SDK_VER=v1.7.1.12 POWER=no SUT=q4_std_edge DOCKER=no MULTISTREAM_ONLY=yes WORKLOADS="resnet50" $(ck find ck-qaic:script:run)/run_edge.sh
```
