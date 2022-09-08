## Benchmark Setup

Setup CM for MLPerf Inference for Resnet50 as detailed [here](/open/OctoML/code/resnet50/README.md)

## Run Command
```
cm run script --tags=app,mlperf,_resnet50,_tf  --output_dir=$HOME/final_results \
--env.CM_SUT_NAME=apple-m1-tf --add_deps_tags.imagenet=_full \
--env.IMAGENET_PATH=$HOME/datasets/imagenet-2012-val --env.CM_LOADGEN_MODE=performance \
--env.CM_LOADGEN_SCENARIO=Offline
```

* Internal command being given to Mlperf Inference Application
```
cd /Users/arjun/CM/repos/local/cache/a1105ad0b3c74f3e/inference/vision/classification_and_detection && \
./run_local.sh tf resnet50 cpu --scenario Offline  --threads 8 --mlperf_conf \
/Users/arjun/CM/repos/local/cache/a1105ad0b3c74f3e/inference/mlperf.conf --user_conf \
/Users/arjun/CM/repos/mlcommons@ck/cm-mlops/script/app-mlperf-inference-vision-reference/tmp/6dfc26388bce4fc0a901362c9cc5ca16.conf
```
