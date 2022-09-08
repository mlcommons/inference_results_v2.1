To run this benchmark, first follow the setup steps in `closed/NVIDIA/README_Triton_AWS_Inferentia.md`.

```
make run_inferentia_harness RUN_ARGS="--benchmarks=resnet50 --scenarios=Offline --config_ver=triton --test_mode=AccuracyOnly"
make run_inferentia_harness RUN_ARGS="--benchmarks=resnet50 --scenarios=Offline --config_ver=triton --test_mode=PerformanceOnly"
```

For more details, please refer to `closed/NVIDIA/README.md`.
