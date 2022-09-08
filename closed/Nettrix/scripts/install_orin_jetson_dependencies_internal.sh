pushd /tmp

# install CUDA
wget --no-check-certificate https://sdkm-a.nvidia.com/builds/SDKManager/JetPack_SDKs/5.0/L4T/93_22033_30946379/JETPACK_50_b93/cuda-repo-l4t-11-4-local_11.4.14-1_arm64.deb -O cuda11.4.deb \
  && sudo dpkg -i ./cuda11.4.deb \
  && sudo apt-key add /var/cuda-repo-l4t-11-4-local/7fa2af80.pub \
  && sudo apt update \
  && sudo apt install -y cuda-toolkit-*

# install CUDNN
wget --no-check-certificate https://sdkm-a.nvidia.com/builds/SDKManager/JetPack_SDKs/5.0/L4T/93_22033_30946379/JETPACK_50_b93/cudnn-local-repo-ubuntu2004-8.3.1.22_1.0-1_arm64.deb -O cudnn8.3.1.deb \
  && sudo dpkg -i ./cudnn8.3.1.deb \
  && sudo apt update \
  && sudo apt install libcudnn8*

# install TRT 8.4
mkdir -p trt
cd trt
wget -np -nd -r http://cuda-repo.nvidia.com/release-candidates/Libraries/TensorRT/v8.4/8.4.0.5-4b984e53/11.4-r470/l4t-aarch64/deb/ \
  && sudo dpkg -i ./libnvinfer8_8.4.0-1+cuda11.4_arm64.deb \
  && sudo dpkg -i ./libnvinfer-dev_8.4.0-1+cuda11.4_arm64.deb \
  && sudo dpkg -i ./libnvinfer-plugin8_8.4.0-1+cuda11.4_arm64.deb \
  && sudo dpkg -i ./libnvinfer-plugin-dev_8.4.0-1+cuda11.4_arm64.deb \
  && sudo dpkg -i ./libnvparsers8_8.4.0-1+cuda11.4_arm64.deb \
  && sudo dpkg -i ./libnvparsers-dev_8.4.0-1+cuda11.4_arm64.deb \
  && sudo dpkg -i ./libnvonnxparsers8_8.4.0-1+cuda11.4_arm64.deb \
  && sudo dpkg -i ./libnvonnxparsers-dev_8.4.0-1+cuda11.4_arm64.deb \
  && sudo dpkg -i ./python3-libnvinfer_8.4.0-1+cuda11.4_arm64.deb \
  && sudo dpkg -i ./python3-libnvinfer-dev_8.4.0-1+cuda11.4_arm64.deb \
  && sudo dpkg -i ./graphsurgeon-tf_8.4.0-1+cuda11.4_arm64.deb \
  && sudo dpkg -i ./uff-converter-tf_8.4.0-1+cuda11.4_arm64.deb
cd ..

popd
