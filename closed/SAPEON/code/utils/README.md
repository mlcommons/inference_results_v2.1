# utils

If you want to use ONNX files for input, run the script?command? below to convert the ONNX file to a binary file which is compatible with SAPEON runtime.  
Wait util the process is terminated even though nothing is printed after the log `Processing images( 499 / 500)..... Done!` printed. The whole process takes about 30 minutes.

```bash
./run.sh |tee -a log.log
```

The results are saved at `./result` by default, and then copied to `/home/shared/weight_result`.

Calibration set used : [https://github.com/mlcommons/inference/blob/master/calibration/ImageNet/cal_image_list_option_1.txt]
