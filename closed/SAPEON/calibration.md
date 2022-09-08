# MLPerf Inference v2.1 - Calibration
Post-training quantization requires a dynamic range for each weight and activation tensor. We use a set of calibration data to compute quantization ranges of the neural network. Per-channel quantization is used for weights, and per-layer quantization is used for activations.
