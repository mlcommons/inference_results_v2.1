# Modularizing MLPerf inference benchmark

This demo submission is a part of several community projects to modularize MLPerf inference benchmark
and automate the submission process across different ML tasks, models, data sets, engines and platforms
using [the 2nd generation of the CK framework (CM)](https://github.com/mlcommons/ck/tree/master/cm):

* [Unifying MLOps and DevOps scripts, tools and artifacts](https://github.com/mlcommons/ck/tree/master/cm-mlops/script)
* [Modularizing MLPerf inference benchmark](https://github.com/mlcommons/ck/issues/261)
* [Implementing modular ML benchmarking based on loadgen and CM](https://github.com/mlcommons/ck/issues/265)

## Reproducibility report

* Install cmind 
```
python3 -m pip install cmind
```

```
cm pull repo:mlcommons@ck
```

* Install the Imagenet dataset
Imagenet is not publically available and we need to obtain it locally. Once we have it in a local folder say `/datasets/imagenet-2012-val`, 
we can register it in CM as follows:

```
cm run script --tags=get,dataset,imagenet,original --env.IMAGENET_PATH=/datasets/imagenet-2012-val
```

* Download MLCommons Inference Source and Install Loadgen

```
cm run script --tags=get,inference,mlcomons,loadgen --version=r2.1
```
* Install patched MLCommons Inference src

To handle [custom OUTPUT_DIR](https://github.com/mlcommons/inference/pull/1207) we need to apply a patch to mlcommons inference src
before doing a run as follows:
```
cm run script --tags=get,inference,mlcomons,loadgen,_patch --version=r2.1
```
