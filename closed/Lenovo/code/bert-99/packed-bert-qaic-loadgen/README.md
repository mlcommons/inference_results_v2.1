# Qualcomm Cloud AI - MLPerf Inference - Language

## Initial System Setup

Complete the common benchmarking setup as detailed [here](https://github.com/krai/ck-qaic/blob/main/program/README.md)


### Install additional Python dependencies (in userspace)

#### Install implicit dependencies via pip

**NB:** These dependencies are _implicit_, i.e. CK will not try to satisfy them. If they are not installed, however, the workflow will fail.

```
export CK_PYTHON=`which python3.8`
${CK_PYTHON} -m pip install --user onnx-simplifier
${CK_PYTHON} -m pip install --user tokenization
${CK_PYTHON} -m pip install --user nvidia-pyindex
${CK_PYTHON} -m pip install --user onnx-graphsurgeon==0.3.11
```

#### Install explicit dependencies via CK (also via `pip`, but register with CK at the same time)

**NB:** These dependencies are _explicit_, i.e. CK will try to satisfy them automatically. On a machine with multiple versions of Python, things can get messy, so we are playing safe here.

```
ck install package --tags=python-package,onnx --force_version=1.8.1
ck install package --tags=lib,python-package,pytorch --force_version=1.8.1 --quiet
ck install package --tags=lib,python-package,transformers --force_version=2.4.0
ck install package --tags=lib,python-package,tensorflow --quiet
```

<a name="prepare_squad_download"></a>
##  Download the SQuAD v1.1 dataset

```
ck install package --tags=dataset,squad,raw,width.384 --quiet
ck install package --tags=dataset,calibration,squad,pickle,width.384 --quiet
```

<a name="prepare_install_model"></a>
##  Install the model

```
ck install package --tags=model,mlperf,qaic,bert-packed --quiet
```

<a name="prepare_calibrate_model"></a>
## Calibrate the model

```
ck install package --tags=profile,qaic,bert-packed --quiet
```

<a name="prepare_compile_workload"></a>
## Compile the workload

### Finding the best PCV Value
The accuracy of the Bert-99 model depends on the Percetile Calibration Value used for compilation. The following script (can take more than an hour) can tell you the best PCV value on a given host system for a given model binary
```
PC_START=80 PC_END=90 $(ck find package:model-qaic-compile)/percentile-calibration.sh bert-99 bert-99.pcie.16nsp.offline offline
```
```
PC_START=80 PC_END=90 $(ck find package:model-qaic-compile)/percentile-calibration.sh bert-99 bert-99.pcie.16nsp.singlestream singlestream
```
The above found best PCV value can be exported to `$PCV` variable for both offline and singlestream
### Compilation for 20w AEDKs (edge category)

```
ck install package --tags=model,compiled,bert-99,bert-99.aedk_20w.offline,quantization.calibration --env._PERCENTILE_CALIBRATION_VALUE=99.$PCV --extra_tags=pcv.$PCV
```
```
ck install package --tags=model,compiled,bert-99,bert-99.aedk_20w.singlestream,quantization.calibration --env._PERCENTILE_CALIBRATION_VALUE=99.$PCV --extra_tags=pcv.$PCV

```

### Compilation for 15w AEDKs (edge category)

```
ck install package --tags=model,compiled,bert-99,bert-99.aedk_15w.offline,quantization.calibration --env._PERCENTILE_CALIBRATION_VALUE=99.$PCV --extra_tags=pcv.$PCV
```
```
ck install package --tags=model,compiled,bert-99,bert-99.aedk_15w.singlestream,quantization.calibration --env._PERCENTILE_CALIBRATION_VALUE=99.$PCV --extra_tags=pcv.$PCV
```

### Compilation for edge category 16 NSP PCIe

```
ck install package --tags=model,compiled,bert-99,bert-99.pcie.16nsp.offline,quantization.calibration --env._PERCENTILE_CALIBRATION_VALUE=99.$PCV --extra_tags=pcv.$PCV
```
```
ck install package --tags=model,compiled,bert-99,bert-99.pcie.16nsp.singlestream,quantization.calibration --env._PERCENTILE_CALIBRATION_VALUE=99.$PCV --extra_tags=pcv.$PCV
```

### Compilation of BERT 99.9% for datacenter category 16 NSP PCIe

```
ck install package --tags=model,compiled,bert-99.9,bert-99.9.pcie.16nsp.offline
```

## Benchmarking
For benchmarking for different System Under Tests, please see [here](https://github.com/krai/ck-qaic/blob/main/program/packed-bert-qaic-loadgen/README.benchmarking.md)

## Info

Please contact anton@krai.ai if you have any problems or questions.
