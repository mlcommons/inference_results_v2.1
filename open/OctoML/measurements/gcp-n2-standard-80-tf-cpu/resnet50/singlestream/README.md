## Benchmark Setup

Setup CM for MLPerf Inference for Resnet50 as detailed [here](/open/OctoML/code/resnet50/README.md)

## Run Command
```
cm run script --tags=app,mlperf,_resnet50,_tf  --output_dir=$HOME/final_results \
--add_deps_tags.imagenet=_full \
--env.IMAGENET_PATH=$HOME/datasets/imagenet-2012-val --env.CM_LOADGEN_MODE=performance \
--env.CM_LOADGEN_SCENARIO=SingleStream --env.CM_SUT_NAME=gcp-n2-standard-80-tf
```

* Internal command being given to Mlperf Inference Application

```
cd /Users/arjun/CM/repos/local/cache/7c3507d84f274c06/inference/vision/classification_and_detection \
&& ./run_local.sh onnxruntime resnet50 cpu --scenario SingleStream --threads 1 --mlperf_conf \
/Users/arjun/CM/repos/local/cache/7c3507d84f274c06/inference/mlperf.conf --user_conf \
/Users/arjun/CM/repos/mlcommons@ck/cm-mlops/script/app-mlperf-inference-vision-reference/tmp/2ad2283cac2f40148d7994f5790d66b0.conf
```
