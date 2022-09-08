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
Follow the languafe/bert build instructions and replace all python files with those provided in the submission

