## Benchmark Setup

Setup CM for MLPerf Inference for Resnet50 as detailed [here](/open/OctoML/code/resnet50/README.md)

### Choose Tensorflow 2.6.0
```
cm run script --tags=get,tensorflow --version=2.6.0
```

## Run Command
```
cm run script --tags=app,mlperf,_resnet50,_tf  --output_dir=$HOME/final_results \
--add_deps_tags.imagenet=_full \
--env.IMAGENET_PATH=$HOME/datasets/imagenet-2012-val --env.CM_LOADGEN_MODE=performance \
--env.CM_LOADGEN_SCENARIO=Offline --env.CM_SUT_NAME=gcp-n2-standard-80-tf
```

* Internal command being given to Mlperf Inference Application

```
cd /Users/arjun/CM/repos/local/cache/7c3507d84f274c06/inference/vision/classification_and_detection \
&& ./run_local.sh onnxruntime resnet50 cpu --scenario Offline --threads 80 --mlperf_conf \
/Users/arjun/CM/repos/local/cache/7c3507d84f274c06/inference/mlperf.conf --user_conf \
/Users/arjun/CM/repos/mlcommons@ck/cm-mlops/script/app-mlperf-inference-vision-reference/tmp/1919119438e1424984a3d68e8f20a66a.conf
```
