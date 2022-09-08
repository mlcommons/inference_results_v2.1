## Benchmark Setup

Setup CM for MLPerf Inference for Retinanet as detailed [here](/open/OctoML/code/retinanet/README.md)

### Install onnx 1.8.1
```
python3 -m pip install onnx==1.8.1
```
## Run Command
```
cm run script --tags=app,mlperf,_resnet50,_onnxruntime  --output_dir=$HOME/final_results \
--env.CM_LOADGEN_MODE=performance --env.CM_LOADGEN_SCENARIO=SingleStream \
--env.CM_SUT_NAME=gcp-n2-standard-80-onnxruntime
```

* Internal command being given to Mlperf Inference Application

```
cd /Users/arjun/CM/repos/local/cache/7c3507d84f274c06/inference/vision/classification_and_detection \
&& ./run_local.sh onnxruntime retinanet cpu --scenario SingleStream  --threads 1 --mlperf_conf \
/Users/arjun/CM/repos/local/cache/7c3507d84f274c06/inference/mlperf.conf --user_conf \
/Users/arjun/CM/repos/mlcommons@ck/cm-mlops/script/app-mlperf-inference-vision-reference/tmp/d70185f8af7146899a7a5b1eaf555f1e.conf
```
