# MLPerf™ Deci-AI Image Classification Results on AMD EPYC MilanX 7773X

The results:

| Submitter | Track | System      | 	Processor                 | Software           | Offline (samples/s) | 	SQuAD v1.1 F1 Score |
|-----------|--------|-----|----------------------------|--------------------|---------------------|------------------------------|
| DELL        | Bert99.9 | DELL Server | 2xAMD EPYC Milan-X 7773X @ | deci (ONNXRuntime) |      111.403       |         91.09              |

## Reproducibility
Due to intellectual property issues we share our benchmarking code and do no disclosed our models. 
To have an interactive session to reproduce the results contact shai.rozenberg@deci.ai


### Prerequisite
Clone the [official mlcommons inference repo](https://github.com/mlcommons/inference/tree/r2.1/language/bert)
Follow the language/bert build instructions and replace all python files with those provided in the submission
### Offline

```
OMP_NUM_THREADS=144 numactl --physcpubind=0-255 --membind 0-1 python run.py --backend=onnxruntime --scenario=Offline --model_path=/home/deci/deciberts/decibert2_quant.onnx --batch_size=10 --intra_threads=32 --inter_threads=4 --e_mode=para --tokenizer=deci```
```

### Accuracy
```
OMP_NUM_THREADS=144 numactl --physcpubind=0-255 --membind 0-1 python run.py --backend=onnxruntime --scenario=Offline --model_path=/home/deci/deciberts/decibert2_quant.onnx --batch_size=10 --intra_threads=32 --inter_threads=4 --e_mode=para --tokenizer=deci -accuracy```
```

