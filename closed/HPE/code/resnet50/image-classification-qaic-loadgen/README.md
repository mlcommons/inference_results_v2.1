# Qualcomm Cloud AI - MLPerf Inference - Image Classification

Please refer to [Docker README](https://github.com/krai/ck-qaic/blob/main/docker/resnet50/README.md) for a faster way to benchmark Image Classification on Qualcomm Cloud AI. The below instructions are meant only when docker environment is not used. 

## Initial System Setup

Complete the common benchmarking setup as detailed [here](https://github.com/krai/ck-qaic/blob/main/program/README.md)

<a name="prepare_imagenet"></a>
## Prepare the ImageNet validation dataset (50,000 images)

<a name="prepare_imagenet_detect"></a>
### Detect

Unfortunately, the ImageNet 2012 validation dataset (50,000 images) [cannot be freely downloaded](https://github.com/mlcommons/inference/issues/542).
If you have a copy of it e.g. under `/datasets/dataset-imagenet-ilsvrc2012-val/`, you can register it with CK ("detect") by giving the absolute path to `ILSVRC2012_val_00000001.JPEG` as follows:

```
echo "full" | ck detect soft:dataset.imagenet.val --extra_tags=ilsvrc2012,full \
--full_path=/datasets/dataset-imagenet-ilsvrc2012-val/ILSVRC2012_val_00000001.JPEG
```

<a name="prepare_imagenet_preprocess"></a>
### Preprocess

**NB:** Since the preprocessed ImageNet dataset takes up 7.1G, you may wish to change its destination directory by appending `--ask` to the below commands.

```
ck install package \
--dep_add_tags.dataset-source=original,full \
--tags=dataset,imagenet,val,full,preprocessed,using-opencv,for.resnet50.quantized,layout.nhwc,side.224,validation
```

<a name="prepare_resnet50"></a>
## Prepare the ResNet50 model

### Download the MLPerf TensorFlow model

```
ck install package --tags=model,tf,mlperf,resnet50,fix_input_shape
```

**NB:** The input tensor's shape gets updated ("fixed") from `?x224x224x3` to `1x224x224x3` to work around a current limitation in the toolchain.


### Obtain a profile using [MLPerf calibration option #1](https://github.com/mlcommons/inference/blob/master/calibration/ImageNet/cal_image_list_option_1.txt)


```
ck install package --dep_add_tags.imagenet-val=full \
--tags=dataset,imagenet,calibration,mlperf.option1
```
```
ck install package --dep_add_tags.dataset-source=mlperf.option1 \
--tags=dataset,preprocessed,using-opencv,for.resnet50,layout.nhwc,first.500 \
--extra_tags=calibration,mlperf.option1
```


#### 8 samples per batch (for the Server and Offline scenarios)

```
ck install package --tags=profile,resnet50,mlperf.option1,bs.8
```

#### 1 sample per batch (for the SingleStream scenario)

```
ck install package --tags=profile,resnet50,mlperf.option1,bs.1
```

## Compile the models
### Compilation for 20w AEDKs (edge category)

```
ck install package \
--dep_add_tags.profile-resnet50=mlperf.option1 \
--tags=model,qaic,resnet50,resnet50.aedk_20w.offline
```
```
ck install package \
--dep_add_tags.profile-resnet50=mlperf.option1 \
--tags=model,qaic,resnet50,resnet50.aedk_20w.singlestream
```
```
ck install package \
--dep_add_tags.profile-resnet50=mlperf.option1 \
--tags=model,qaic,resnet50,resnet50.aedk_20w.multistream
```

### Compilation for 15w AEDKs (edge category)

```
ck install package \
--dep_add_tags.profile-resnet50=mlperf.option1 \
--tags=model,qaic,resnet50,resnet50.aedk_15w.offline
```
```
ck install package \
--dep_add_tags.profile-resnet50=mlperf.option1 \
--tags=model,qaic,resnet50,resnet50.aedk_15w.singlestream
```
```
ck install package \
--dep_add_tags.profile-resnet50=mlperf.option1 \
--tags=model,qaic,resnet50,resnet50.aedk_15w.multistream
```

Once models are compiled for AEDKs they can be installed on to the device(s) using [this](https://github.com/krai/ck-qaic/tree/main/script/setup.aedk#hr-compile-the-models-and-copy-to-the-device) script.


### Compilation for edge category 16 NSP PCIe

```
ck install package \
--dep_add_tags.profile-resnet50=mlperf.option1 \
--tags=model,qaic,resnet50,resnet50.pcie.16nsp.offline
```
```
ck install package \
--dep_add_tags.profile-resnet50=mlperf.option1 \
--tags=model,qaic,resnet50,resnet50.pcie.16nsp.singlestream
```
```
ck install package \
--dep_add_tags.profile-resnet50=mlperf.option1 \
--tags=model,qaic,resnet50,resnet50.pcie.16nsp.multistream
```

### Compilation for datacenter category 16 NSP PCIe

```
ck install package \
--dep_add_tags.profile-resnet50=mlperf.option1 \
--tags=model,qaic,resnet50,resnet50.pcie.16nsp.offline
```

### Benchmarking
For benchmarking for different System Under Tests, please see [here](https://github.com/krai/ck-qaic/blob/main/program/image-classification-qaic-loadgen/README.benchmarking.md)

## Info

Please contact anton@krai.ai if you have any problems or questions.
