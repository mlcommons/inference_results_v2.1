## Benchmark Setup

Setup CM for MLPerf Inference for Resnet50 as detailed [here](/open/OctoML/code/resnet50/README.md) 

## Run Command
```
cm run script --tags=app,mlperf,_resnet50,_onnxruntime  --output_dir=$HOME/final_results \
--env.CM_SUT_NAME=apple-m1-onnxruntime --add_deps_tags.imagenet=_full \
--env.IMAGENET_PATH=$HOME/datasets/imagenet-2012-val --env.CM_LOADGEN_MODE=performance \
--env.CM_LOADGEN_SCENARIO=Offline
```

* Internal command being given to Mlperf Inference Application

```
cd /Users/arjun/CM/repos/local/cache/a1105ad0b3c74f3e/inference/vision/classification_and_detection \
&& ./run_local.sh onnxruntime resnet50 cpu --scenario Offline  --threads 8 --mlperf_conf \
/Users/arjun/CM/repos/local/cache/a1105ad0b3c74f3e/inference/mlperf.conf --user_conf \
/Users/arjun/CM/repos/mlcommons@ck/cm-mlops/script/app-mlperf-inference-vision-reference/tmp/501612d4a6b840d7889fef65542840e5.conf
```
