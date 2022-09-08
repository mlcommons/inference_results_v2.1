Birentech MLPerf Quantization  

Birentech adopts Post-Training Quantization solution and dynamic-range symmetric quantization to quantize weights, activation tensors from FP32 to int8. 

Weights 
Weight is usually quantized in channel-wise.  
First: find the range of weights for every layer: [min, max] 
Second: calculate the scalar of quantization on weight : (max-min) / 256 
Third: if symmetric quantization, zero-point is 0.  So,  quantized weight = rounding ( weight / scalar + zero-point )  

Activations 
Birentech adopts layer-wise quantization for activation tensors. We only calibrate activation tensors with a scalar by computing 300 samples for image classification model but others. 
First: find the range of activations for every layer: [min, max] 
Second: calculate the scalar of quantization on activation : (max-min) / 256 
Third: if symmetric quantization, zero-point is 0.  So,  quantized activation = rounding ( activation / scalar + zero-point )  
But for first layer in ResNet50, we use non-symmetric quantization and set zero-point to 115 statistically but others. 