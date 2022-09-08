# MLPerf Inference v2.1 Alibaba SINIAN VODLA Implementations
This is a repository of Alibaba-optimized implementations for the [MLPerf](https://mlcommons.org/en/) Inference Benchmark.
It is based on the implementations from [NVIDIA](https://github.com/mlcommons/inference_results_v2.0/tree/master/closed/NVIDIA). 
Pleaese refer to the [README.md](https://github.com/mlcommons/inference_results_v2.0/blob/master/README.md) for environment setup and model download.  

---

### Launching the server

In docker container, run

```
$ odla_tensorrt_grpc_daemon
```

### Launching the client

In docker container, run the make command. For example:

```
make run RUN_ARGS="--benchmarks=bert --scenarios=Server --devices=16 --system_name=SINIAN_VODLA_EFLO_A30x16"
make run_harness RUN_ARGS="--benchmarks=bert --scenarios=Server --devices=16 --system_name=SINIAN_VODLA_EFLO_A30x16"
make update_results
```

