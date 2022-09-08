To run this benchmark, first follow the setup steps in `closed/NVIDIA/README_Triton_AWS_Inferentia.md`.

```
make run_inferentia_harness RUN_ARGS="--benchmarks=bert --scenarios=Offline --config_ver=high_accuracy_triton --test_mode=AccuracyOnly"
make run_inferentia_harness RUN_ARGS="--benchmarks=bert --scenarios=Offline --config_ver=high_accuracy_triton --test_mode=PerformanceOnly"
```

For more details, please refer to `closed/NVIDIA/README.md`.