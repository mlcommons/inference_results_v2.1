This README is a quickstart guide on how to use our code as a user to run testing on Biren's pre-setup servers.

If something is confusing in this Document, please contact Biren.

### Biren's Submission

Biren submits with only one system with 8 GPUs in datacenter category, however it also supports to run on server with 1, 2 and 4 GPUs.

Our submission implements 2 inference harnesses stored under closed/Biren/code/harness.
    - default harness - For Resnet50
    - bert harness

### System Requirements and Software Dependencies

Our submission works with Biren's GPU cards and uses Docker to set up the environment. Requirements are:

- [Docker CE](https://docs.docker.com/engine/install/)
    - If you have issues with running Docker without sudo, follow this [Docker guide from DigitalOcean](https://www.digitalocean.com/community/questions/how-to-fix-docker-got-permission-denied-while-trying-to-connect-to-the-docker-daemon-socket) on how to enable Docker for your new non-root user. Namely, add your new user to the Docker usergroup, and remove ~/.docker or chown it to your new user.
    - You may also have to restart the docker daemon for the changes to take effect:

      `$ sudo systemctl restart docker `

### Setting up the Scratch Spaces

Biren's MLPerf Inference submission stores the models, datasets, preprocessed datasets and kernels in a central location we refer to as a "Scratch Space".

Because of the large amount of data that needs to be stored in the scratch space, we recommend that the scratch be at least **1 TB**. This size is recommended to obtain all datasets in order to run each benchmark and have extra room to store logs, kernels etc.

**Note that once the scratch space is setup and all the data, models, preprocessed datasets and kernels are set up, you do not have to re-run this step.** Users will only need to revisit this step if:

- Users accidentally corrupted or deleted your scratch space
- Users need to redo the steps for a benchmark you previously did not need to set up
- Users has decided that something in the preprocessing step needed to be altered

Once you have obtained a scratch space, set the `MLPERF_SCRATCH_PATH` environment variable. This is how our code tracks where the data is stored. By default, if this environment variable is not set, we assume the scratch space is located at `/home/mlperf/workspace/data`.

Then create empty directories in your scratch space to house the data:

`$ mkdir $MLPERF_SCRATCH_PATH/data $MLPERF_SCRATCH_PATH/models $MLPERF_SCRATCH_PATH/preprocessed_data`

Next, export this path as environment variable.

`$ export MLPERF_SCRATCH_PATH=/path/to/scratch/space`

Folders in this `MLPERF_SCRATCH_PATH` will also be mounted inside the docker container under the folder /work/build.

After you have done so, you will need to download the models and datasets, and run the preprocessing scripts on the datasets.

Put Biren full-stack package under closed/Biren/docker/packages. Enter the container by entering the `closed/Biren` directory and running:

`make biren_docker_test FULL_STACK_PATH=$FULL_STACK_NAME # Build, launch docker and mount data, models and preprocessed_data folder into /work/build/`

### Download the Datasets

Each benchmark contains a script to download dataset (located at `closed/Biren/code/[benchmark name]/dataProcess/download_data.sh`) that downloads or explains how to download the datasets for that benchmark. However, you do not need to actually use that script to download dataset.

### Downloading the Model files

Each benchmark contains a script to download models (located at `closed/Biren/code/[benchmark name]/modelProcess/download_model.sh`) that downloads or explains how to download the model files for that benchmark. However, you do not need to actually use that script to download models. You can manually download the files from the MLCommons Github. The MLCommons Inference committee curates [a list of links](https://github.com/mlcommons/inference/blob/master/README.md) to the reference models that all MLPerf Inference submitters are required to use.


### Process Datasets and models

#### resnet50

Preprocess dataset:
  - move downloaded dataset to folder `/work/build/preprocessed_data/imagenet/`

  - go to dataProcess folder and process dataset
    
    build data process project

    - `cd /work/code/resnet50/dataProcess/`
    - `mkdir build`
    - `cd build`
    - `cmake ..`
    - `make`

    execute the program to process images

    - `./suinfer_preprocess`

Preprocess model:
  - copy downloaded model resnet50_v1.onnx to folder `code/resnet50/modelProcess/inference_calibration`

  - preprocess model
  
    `cd /work/code/resnet50/modelProcess/inference_calibration && bash rn50_cali.sh`

  - move preprocessed model to models folder `/work/build/models/resnet50`

    `mv resnet50_int8_delete_quantize_linear_v2.onnx /work/build/models/resnet50`

#### bert_99.9
Preprocess dataset:
  - copy dataset to folder `/work/code/bert-99.9/dataProcess`

  - go to dataProcess folder and process dataset
  
    `cd /work/code/bert-99.9/dataProcess && ./start_convert.sh`

  - move preprocess dataset to `/work/build/preprocessed_data/bert_data_int32/`

    `mv input_ids /work/build/preprocessed_data/bert_data_int32/`

    `mv input_mask /work/build/preprocessed_data/bert_data_int32/`

    `mv segment_ids /work/build/preprocessed_data/bert_data_int32/`

Preprocess model:
  - copy downloaded model bert_large_v1_1.onnx to folder `/work/code/bert-99.9/modelProcess`

  - preprocess model
  
    `cd /work/code/bert-99.9/modelProcess && python bert_99.9_preprocess.py`

  - move preprocessed model to models folder

    `mv 4inputs_whole_graph.onnx /work/build/models/bert/bert_large_99.9_4inputs.onnx`

### install python dependencies
`pip install colorlog xlrd==1.2.0 openpyxl numpy torch termcolor pytest bitstring mako pyyaml six`

### build all deps in docker, for examples, the loadgen
`make` or `make biren`

#### run resnet50 test, systemid can be BR104-300W-PCIex8, BR104-300W-PCIex4, BR104-300W-PCIex2 or BR104-300W-PCIex1
`rn50.sh [accu|perf] [Offline|Server] systemid`

#### run bert-99.9 test, systemid can be BR104-300W-PCIex8, BR104-300W-PCIex4, BR104-300W-PCIex2 or BR104-300W-PCIex1
`bert_99.9.sh [accu|perf] [Offline|Server] systemid`
