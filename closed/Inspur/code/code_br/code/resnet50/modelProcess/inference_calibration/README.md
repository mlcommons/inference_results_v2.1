# inference_calibration
MLPerf Inference Calibration 

We use Intel neural-compressor library as one of dependencies. Thanks a lot for the contribution.

And we modify some source code in March,2022 as needed under Apache License Version 2.0.

------
The modified file list as below:

examples/onnxrt/image_recognition/onnx_model_zoo/resnet50/quantization/ptq/resnet50_v1_5.yaml

examples/onnxrt/image_recognition/resnet50/quantization/ptq/main.py

neural_compressor/adaptor/onnxrt.py

neural_compressor/adaptor/ox_utils/onnx_quantizer.py

neural_compressor/adaptor/ox_utils/util.py

neural_compressor/experimental/metric/metric.py

------

## Dependencies
- python3
- Intel neural-compressor

## Run
bash rn50_cali.sh

------

Copyright 2022 Birentech

Licensed under the Apache License, Version 2.0 (the "License")