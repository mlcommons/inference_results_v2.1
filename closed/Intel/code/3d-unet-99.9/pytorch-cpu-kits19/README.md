# MLPerf Inference Benchmarks for Medical Image 3D Segmentation

## Steps to run 3D-UNet


### HW and SW requirements
```
  SPR 2 sockets
  GCC >= 11.2
```

## Run on host

### 1. Install Anaconda 3.0

```
  mkdir <workfolder>
  cd <workfolder>
  wget -c https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh -O anaconda3.sh
  chmod +x anaconda3.sh
  ./anaconda3.sh -b -p ./anaconda3
  export WORKDIR=$PWD
  export PATH=$WORKDIR/anaconda3/bin:$PATH
  conda create -n 3dunet python=3.8
  conda update -n base -c defaults conda --yes
  source activate 3dunet
```

### 2. Download Dataset
```
  cd ~
  mkdir -p mlperf_data/3dunet-kits
  cd mlperf_data/3dunet-kits
  git clone https://github.com/neheller/kits19
  cd kits19
  pip3 install -r requirements.txt
  python3 -m starter_code.get_imaging
  export DOWNLOAD_DATA_DIR=${PWD}/data
  cd ~
```

### 3. Download Repo for 3D-UNet MLPerf inference and setup env
```
  git clone <this/repo>
  // REPO_PATH=<path/to/this/repo>
  bash REPO_NAME/closed/Intel/code/3d-unet-99.9/pytorch-cpu-kits19/prepare_env.sh --code=REPO_PATH
```

### 4. Run command for accuracy and performance
```
  cd REPO_NAME/closed/Intel/code/3d-unet99.9/pytorch-cpu-kits19
  bash run_SPR56C_2S.sh acc
  bash run_SPR56C_2S.sh perf
```



## Run on docker

### 1. Download Dataset on host

```
follow the #step2-Download Dataset to prepare dataset on host
```

###  2. start a container and login

```
docker run --name intel_3dunet --privileged -itd --net=host --ipc=host -v </path/of/3dunet/above>:/data/mlperf_data/3dunet intel/intel-optimized-pytorch:mlperf-submission-inference-2.1-3dunet
 #check
 docker ps
 #into container
 docker exec -it intel_3dunet bash
```

###  3. run

```
cd code/3d-unet-99.9/pytorch-cpu-kits19
export http_proxy=<your/http_proxy>
export https_proxy=<your/https_proxy>
#make preprocess
bash make_preprocess.sh

#run commad
bash run_SPR56C_2S.sh acc
bash run_SPR56C_2S.sh perf
```

