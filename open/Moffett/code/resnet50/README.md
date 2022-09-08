Moffett MLCommons v2.1 Inference Workspace
======

# Target Scenarios
- [x] RESNET50 Offline
- [x] BERT-LARGE Offline

# Data Preparation
| Directory  | Location                          | Description                                                 |
|------------|-----------------------------------|-------------------------------------------------------------|
| Dataset    | /home/moffett/mlcommons/datasets  | ImageNet 2012 / SQuAD v1.1 validation set                   |
| Model      | /home/moffett/mlcommons/models    | Compressed RN50/BERT-LARGE MODELS                           |
| Library    | /home/moffett/mlcommons/library   | Moffett Shared Library                                      |                   |                          |
| Runtime    | /home/moffett/mlcommons/runtime   | Moffett Runtime                                             |
| Inference  | /home/moffett/mlcommons/inference | Repository https://github.com/mlcommons/inference/tree/r2.1 |                                       |

# Environment Preparation
## Install Loadgen   
```bash
sudo su
source /home/moffett/mlcommons/init_conda.sh
apt-get install libglib2.0-dev pybind11-dev
pip3 install absl-py numpy
cd /home/moffett/mlcommons/inference/loadgen
CFLAGS="-std=c++14 -O3" python setup.py bdist_wheel
pip install --force-reinstall dist/*.whl
```

# Set Hardware Platform
- S4：export HARDWARE_PLATFORM=s4
- S10: export HARDWARE_PLATFORM=s10
- S30：export HARDWARE_PLATFORM=s30

# Run Resnet50 Offline Tests

## AccuracyOnly Mode
```bash
cd sut/Offline/resnet50
python3 run_resnet50.py --config ../../../config/resnet50_offline_${HARDWARE_PLATFORM}.yaml --mode AccuracyOnly --output_dir accuracy
```

## Check Accuracy
```bash
python3 accuracy-imagenet.py --mlperf-accuracy-file accuracy/mlperf_log_accuracy.json --imagenet-val-file ../../../../data_maps/imagenet/val_map.txt --dtype int32
```

## PerformanceOnly Mode
```bash
python3 run_resnet50.py --config ../../../config/resnet50_offline_${HARDWARE_PLATFORM}.yaml --mode PerformanceOnly --output_dir performance/run_1
```

## Run Resnet50 Offline Audit Test
```bash
bash ./qa-audit_resnet50.sh
```

**Note:** *accuracy-imagenet.py* is
in [inference/vision/classification_and_detection/tools/accuracy-imagenet.py](https://github.com/mlcommons/inference/blob/r2.1/vision/classification_and_detection/tools/accuracy-imagenet.py) 

# Run BERT Offline Tests

## AccuracyOnly Mode
```bash
cd sut/Offline/bert
python3 run_bert.py --config ../../../config/bert_offline_${HARDWARE_PLATFORM}.yaml --mode AccuracyOnly --output_dir accuracy
```

## Check Accuracy
```bash
python accuracy-squad.py --vocab_file /home/moffett/mlcommons/datasets/bert/vocab.txt --val_data /home/moffett/mlcommons/datasets/bert/dev-v1.1.json --log_file accuracy/mlperf_log_accuracy.json --out_file accuracy/predictions.json
```

## PerformanceOnly Mode
```bash
python3 run_bert.py --config  ../../../config/bert_offline_${HARDWARE_PLATFORM}.yaml --mode PerformanceOnly --output_dir performance/run_1
```

## Run BERT Offline Audit Test
```bash
bash ./qa-audit_bert.sh
```

**Note:** *accuracy-squad.py* is
in [inference/language/bert/accuracy-squad.py](https://github.com/mlcommons/inference/blob/r2.1/language/bert/accuracy-squad.py) 
