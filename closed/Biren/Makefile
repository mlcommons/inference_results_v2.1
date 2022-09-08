# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright © 2022 Shanghai Biren Technology Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

SHELL := /bin/bash
MAKEFILE_NAME := $(lastword $(MAKEFILE_LIST))

ARCH := $(shell uname -p)
UNAME := $(shell whoami)
UID := $(shell id -u `whoami`)
HOSTNAME := $(shell hostname)
GROUPNAME := $(shell id -gn `whoami`)
GROUPID := $(shell id -g `whoami`)
TIMESTAMP := $(shell date +'%Y.%m.%d-%H.%M.%S')
ifndef HOST_HOSTNAME
    HOST_HOSTNAME := $(HOSTNAME)
endif
HOST_HOSTNAME_SHORT := $(firstword $(subst ., , $(HOST_HOSTNAME)))
HOSTNAME_SHORT := $(firstword $(subst ., , $(HOSTNAME)))
PYTHON3_CMD := $(shell if ! python3 --version | grep 3.8 &> /dev/null; then echo python3.8; else echo python3; fi)

SUBMITTER ?= BIREN

# Arch and board variables.
SYSTEM_NAME ?= $(shell $(PYTHON3_CMD) scripts/get_system_id.py 2> /dev/null)
TARGET_X86_64 := 0
IS_XAVIER := 0
ifeq ($(ARCH), x86_64)
    TARGET_X86_64 = 1
endif

# Docker is supported for all non-xavier x86/aarch64 system.
SUPPORT_DOCKER := 1
ifeq ($(ARCH), $(filter $(ARCH), x86_64 aarch64))
ifneq ($(IS_XAVIER), 1)
    SUPPORT_DOCKER = 1
endif
endif

# Conditional Docker flags
ifndef DOCKER_DETACH
    DOCKER_DETACH := 0
endif
ifndef DOCKER_TAG
    DOCKER_TAG := $(UNAME)-$(ARCH)
endif

DOCKER_BUILDKIT ?= 1

BIREN_DOCKER_NAME := mlperf-inference-biren-$(DOCKER_TAG)

PROJECT_ROOT := $(shell pwd)
BUILD_DIR    := $(PROJECT_ROOT)/build

HOST_VOL ?= ${PWD}
CONTAINER_VOL ?= /work
NO_DOCKER_PULL ?= 0
NO_BUILD ?= 0

# Set the include directory for suinfer header files
SUINFER_DIR = $(BUILD_DIR)/suInfer
SUINFER_BUILD_DIR = $(SUINFER_DIR)/build

# Set the include directory for Loadgen header files
INFERENCE_DIR = $(BUILD_DIR)/inference
INFERENCE_URL = https://github.com/mlcommons/inference.git
LOADGEN_INCLUDE_DIR ?= $(INFERENCE_DIR)/loadgen
LOADGEN_LIB_DIR ?= $(LOADGEN_INCLUDE_DIR)/build
INFERENCE_HASH = 5dd9ed739cc8dfeae2c5d30ec06f4cb524a4dd74

# Set Environment variables to extracted contents
export LD_LIBRARY_PATH := $(LD_LIBRARY_PATH):/usr/lib/$(ARCH)-linux-gnu:$(LOADGEN_LIB_DIR)
export LIBRARY_PATH := /usr/lib/$(ARCH)-linux-gnu:$(LOADGEN_LIB_DIR):$(LIBRARY_PATH)
export CPATH := /usr/include/$(ARCH)-linux-gnu:/usr/include/$(ARCH)-linux-gnu/cub:$(CPATH)
export PYTHONPATH := $(LOADGEN_LIB_DIR):$(PYTHONPATH)

# Set DATA_DIR, PREPROCESSED_DATA_DIR, and MODEL_DIR if they are not already set
ifndef DATA_DIR
    export DATA_DIR := $(BUILD_DIR)/data
endif
ifndef PREPROCESSED_DATA_DIR
    export PREPROCESSED_DATA_DIR := $(BUILD_DIR)/preprocessed_data
endif
ifndef MODEL_DIR
    export MODEL_DIR := $(BUILD_DIR)/models
endif

# Please run `export MLPERF_SCRATCH_PATH=<path>` to set your scratch space path.
# The below paths are for internal use only.
ifneq ($(wildcard /home/scratch.mlperf_inference),)
    MLPERF_SCRATCH_PATH ?= /home/scratch.mlperf_inference
endif
# Please run `export MLPERF_CPU_SCRATCH_PATH=<path>` to set your scratch space path.
# The below paths are for internal use only.
ifneq ($(wildcard /home/scratch.svc_compute_arch),)
    DOCKER_MOUNTS += -v /home/scratch.svc_compute_arch:/home/scratch.svc_compute_arch
    DOCKER_MOUNTS += -v /home/scratch.computelab/sudo:/home/scratch.computelab/sudo
endif
ifneq ($(wildcard /home/scratch.dlsim),)
    DOCKER_MOUNTS += -v /home/scratch.dlsim:/home/scratch.dlsim
endif
ifneq ($(wildcard $(PROJECT_ROOT)/../../regression),)
    DOCKER_MOUNTS += -v $(PROJECT_ROOT)/../../regression:/regression
endif
ifdef MLPERF_SCRATCH_PATH
    ifneq ($(wildcard $(MLPERF_SCRATCH_PATH)),)
        DOCKER_MOUNTS += -v $(MLPERF_SCRATCH_PATH):$(MLPERF_SCRATCH_PATH)
    endif
endif

# Specify default dir for harness output logs.
ifndef LOG_DIR
    export LOG_DIR := $(BUILD_DIR)/logs/$(TIMESTAMP)
    export POWER_LOGS_DIR := $(BUILD_DIR)/power_logs/$(TIMESTAMP)
endif

# Specify debug options for build (default to Release build)
ifeq ($(DEBUG),1)
    BUILD_TYPE := Debug
else
    BUILD_TYPE := Release
endif

# Handle different nvidia-docker version. Do not use nvidia-docker when running with CPUs
DOCKER_RUN_CMD := docker run

# If DOCKER_COMMAND is not pass, launch interactive docker container session.
ifeq ($(DOCKER_COMMAND),)
    DOCKER_INTERACTIVE_FLAGS = -it
else
    DOCKER_INTERACTIVE_FLAGS =
endif

BASE_IMAGE ?= ubuntu:18.04

############################## BIREN ##############################
.PHONY: biren
biren:
	@$(MAKE) -f $(MAKEFILE_NAME) biren_build

.PHONY: biren_build
biren_build: clone_loadgen link_dirs
	@$(MAKE) -f $(MAKEFILE_NAME) build_loadgen
	@$(MAKE) -f $(MAKEFILE_NAME) build_harness

.PHONY: biren_docker
biren_docker:
	@$(MAKE) -f $(MAKEFILE_NAME) biren_build_docker_test NO_BUILD?=1
ifneq ($(strip ${DOCKER_DETACH}), 1)
	@$(MAKE) -f $(MAKEFILE_NAME) attach_docker || true
endif

FULL_STACK_FILENAME := ""

.PHONY: biren_docker_test
biren_docker_test:
	@$(MAKE) -f $(MAKEFILE_NAME) biren_build_docker_test NO_BUILD?=1 TEST_MODE_NAME?=.test 
ifneq ($(strip ${DOCKER_DETACH}), 1)
	@$(MAKE) -f $(MAKEFILE_NAME) attach_docker || true
endif

.PHONY: biren_docker_dev
biren_docker_dev:
	@$(MAKE) -f $(MAKEFILE_NAME) biren_docker

.PHONY: biren_build_docker_test
biren_build_docker_test:
	@echo $(FULL_STACK_FILENAME)
	docker pull $(BASE_IMAGE)
	DOCKER_BUILDKIT=$(DOCKER_BUILDKIT) docker build -t mlperf-inference-biren:$(DOCKER_TAG)-latest \
		--build-arg BASE_IMAGE=$(BASE_IMAGE) \
		--build-arg FULL_STACK_PATH=packages/$(FULL_STACK_FILENAME) \
		--network host \
		-f docker/Dockerfile$(TEST_MODE_NAME).$(ARCH) docker

TEST_MODE :=""
SCENARIO :=""
SYSTEMID :=""
TEST_CASES :=""
RUN_ARGS :=""

.PHONY: test_rn50 
test_rn50: link_dirs
	@$(SHELL) rn50.sh $(TEST_MODE) $(SCENARIO) $(SYSTEMID)

.PHONY: test_bert_large_999
test_bert_large_999:
	@$(SHELL) bert_99.9.sh $(TEST_MODE) $(SCENARIO) $(SYSTEMID)

.PHONY: test_bert_large_99
test_bert_large_99:
	@$(SHELL) bert_99.sh $(TEST_MODE) $(SCENARIO) $(SYSTEMID)

BIREN_DRIVER_PATH := "/home/br104/umd/br_kmd/biren.ko"
.PHONY: uninstall_biren_driver
uninstall_biren_driver: 
	-sudo rmmod biren

Offline_VAR := "Offline"

.PHONY: auto_test_rn50
auto_test_rn50: uninstall_biren_driver link_dirs 
	@echo "auto_test_rn50 no enter docker"
ifeq ($(SCENARIO), Offline) 
	-sudo insmod $(BIREN_DRIVER_PATH) biren_l2pad_size=2
else
	-sudo insmod $(BIREN_DRIVER_PATH) biren_dtg_osd_num=0
endif
	@$(MAKE) -f $(MAKEFILE_NAME) biren_build_docker_test NO_BUILD?=1 TEST_MODE_NAME?=.test 
	@$(MAKE) -f $(MAKEFILE_NAME) docker_add_user
	@$(MAKE) -f $(MAKEFILE_NAME) no_interactive_launch_docker BASH_NAME?=rn50.sh

.PHONY: auto_test_bert_999
auto_test_bert_999: uninstall_biren_driver link_dirs
	@echo "auto_test_bert_999 no enter docker"
	sudo insmod $(BIREN_DRIVER_PATH) biren_l2pad_size=2
	@$(MAKE) -f $(MAKEFILE_NAME) docker_add_user
	@$(MAKE) -f $(MAKEFILE_NAME) no_interactive_launch_docker BASH_NAME?=bert_99.9.sh
	
.PHONY: auto_test_bert_99
auto_test_bert_99: uninstall_biren_driver link_dirs
	@echo "auto_test_bert_99 no enter docker"
	sudo insmod $(BIREN_DRIVER_PATH) biren_l2pad_size=2
	@$(MAKE) -f $(MAKEFILE_NAME) docker_add_user
	@$(MAKE) -f $(MAKEFILE_NAME) no_interactive_launch_docker BASH_NAME?=bert_99.sh

.PHONY: no_interactive_launch_docker
no_interactive_launch_docker:
	$(DOCKER_RUN_CMD) --rm -w /work \
		-v $(HOST_VOL):$(CONTAINER_VOL) \
		-v /home/$(UNAME):/home/$(UNAME) \
		--privileged=true \
		-v /dev/biren:/dev/biren:rw \
		-v /home/br104/umd:/home/br104/umd \
		--cap-add SYS_ADMIN --cap-add SYS_TIME \
		--shm-size=32gb \
		-v /etc/timezone:/etc/timezone:ro -v /etc/localtime:/etc/localtime:ro \
		--security-opt apparmor=unconfined --security-opt seccomp=unconfined \
		--name $(BIREN_DOCKER_NAME) -h $(shell echo $(BIREN_DOCKER_NAME) | cut -c -64) --add-host $(BIREN_DOCKER_NAME):127.0.0.1 \
		--user $(UID):$(GROUPID) --net host \
		$(DOCKER_MOUNTS) $(DOCKER_ARGS) \
		-e MLPERF_SCRATCH_PATH=$(MLPERF_SCRATCH_PATH) \
		-e HOST_HOSTNAME=$(HOSTNAME) \
		mlperf-inference-biren:$(DOCKER_TAG) $(DOCKER_COMMAND) bash -c "/work/scripts/biren/run_test_without_docker.sh $(BASH_NAME) $(TEST_MODE) $(SCENARIO) $(SYSTEMID)"

.PHONY: no_interactive_launch_docker_compliance
no_interactive_launch_docker_compliance:
	$(DOCKER_RUN_CMD) --rm -w /work \
		-v $(HOST_VOL):$(CONTAINER_VOL) \
		-v /home/$(UNAME):/home/$(UNAME) \
		--privileged=true \
		-v /dev/biren:/dev/biren:rw \
		-v /home/br104/umd:/home/br104/umd \
		--cap-add SYS_ADMIN --cap-add SYS_TIME \
		--shm-size=32gb \
		-v /etc/timezone:/etc/timezone:ro -v /etc/localtime:/etc/localtime:ro \
		--security-opt apparmor=unconfined --security-opt seccomp=unconfined \
		--name $(BIREN_DOCKER_NAME) -h $(shell echo $(BIREN_DOCKER_NAME) | cut -c -64) --add-host $(BIREN_DOCKER_NAME):127.0.0.1 \
		--user $(UID):$(GROUPID) --net host \
		$(DOCKER_MOUNTS) $(DOCKER_ARGS) \
		-e MLPERF_SCRATCH_PATH=$(MLPERF_SCRATCH_PATH) \
		-e HOST_HOSTNAME=$(HOSTNAME) \
		mlperf-inference-biren:$(DOCKER_TAG) $(DOCKER_COMMAND) bash -c "bash $(BASH_NAME) $(SYSTEMID) $(SCENARIO) $(TEST_CASES)"

.PHONY: bert_999_compliance_test_all
bert_999_compliance_test_all: uninstall_biren_driver link_dirs
	@echo "bert_999_compliance_test"
	sudo insmod $(BIREN_DRIVER_PATH) biren_l2pad_size=2
	@$(MAKE) -f $(MAKEFILE_NAME) docker_add_user
	@$(MAKE) -f $(MAKEFILE_NAME) no_interactive_launch_docker_compliance BASH_NAME?=bert_99.9_compliance_run.sh SYSTEMID?=BR104-300W-PCIex8,BR104-300W-PCIex4 SCENARIO?=Offline,Server TEST_CASES?=TEST01,TEST05

.PHONY: bert_999_compliance_test
bert_999_compliance_test: uninstall_biren_driver link_dirs
	@echo "bert_999_compliance_test"
	sudo insmod $(BIREN_DRIVER_PATH) biren_l2pad_size=2
	@$(MAKE) -f $(MAKEFILE_NAME) docker_add_user
	@$(MAKE) -f $(MAKEFILE_NAME) no_interactive_launch_docker_compliance BASH_NAME?=bert_99.9_compliance_run.sh


.PHONY: rn50_compliance_test_server
rn50_compliance_test_server: uninstall_biren_driver link_dirs
	@echo "rn50_compliance_test_all"
	sudo insmod $(BIREN_DRIVER_PATH) biren_dtg_osd_num=0
	@$(MAKE) -f $(MAKEFILE_NAME) docker_add_user
	@$(MAKE) -f $(MAKEFILE_NAME) no_interactive_launch_docker_compliance BASH_NAME?=rn50_compliance_run.sh SYSTEMID?=BR104-300W-PCIex8,BR104-300W-PCIex4 SCENARIO?=Server TEST_CASES?=TEST01,TEST04,TEST05

.PHONY: rn50_compliance_test_offline
rn50_compliance_test_offline: uninstall_biren_driver link_dirs
	@echo "rn50_compliance_test_all"
	sudo insmod $(BIREN_DRIVER_PATH)
	@$(MAKE) -f $(MAKEFILE_NAME) docker_add_user
	@$(MAKE) -f $(MAKEFILE_NAME) no_interactive_launch_docker_compliance BASH_NAME?=rn50_compliance_run.sh SYSTEMID?=BR104-300W-PCIex8,BR104-300W-PCIex4 SCENARIO?=Offline TEST_CASES?=TEST01,TEST04,TEST05

.PHONY: rn50_compliance_test_all
rn50_compliance_test_all: uninstall_biren_driver link_dirs
	@echo "rn50_compliance_test_all"
	@$(MAKE) -f $(MAKEFILE_NAME) docker_add_user
	@$(MAKE) -f $(MAKEFILE_NAME) rn50_compliance_test_server
	@$(MAKE) -f $(MAKEFILE_NAME) rn50_compliance_test_offline

.PHONY: rn50_compliance_test
rn50_compliance_test: uninstall_biren_driver link_dirs
	@echo "rn50_compliance_test"
	@$(MAKE) -f $(MAKEFILE_NAME) docker_add_user
ifeq ($(SCENARIO), Offline) 
	@$(MAKE) -f $(MAKEFILE_NAME) rn50_compliance_test_offline
else
	@$(MAKE) -f $(MAKEFILE_NAME) rn50_compliance_test_server
endif

# Add symbolic links to scratch path if it exists.BR104-300W-PCIex8,BR104-300W-PCIex4 Offline,Server TEST01,TEST04,TEST05
.PHONY: link_dirs
link_dirs:
	@mkdir -p build
ifdef MLPERF_SCRATCH_PATH
	@mkdir -p $(MLPERF_SCRATCH_PATH)/data
	@mkdir -p $(MLPERF_SCRATCH_PATH)/preprocessed_data
	@mkdir -p $(MLPERF_SCRATCH_PATH)/models
	@ln -sfn $(MLPERF_SCRATCH_PATH)/data $(DATA_DIR)
	@ln -sfn $(MLPERF_SCRATCH_PATH)/preprocessed_data $(PREPROCESSED_DATA_DIR)
	@ln -sfn $(MLPERF_SCRATCH_PATH)/models $(MODEL_DIR)
endif

# Add current user into docker image.
.PHONY: docker_add_user
docker_add_user:
ifeq ($(SUPPORT_DOCKER), 1)
	@echo "Adding user account into image"
	DOCKER_BUILDKIT=$(DOCKER_BUILDKIT) docker build -t mlperf-inference-biren:$(DOCKER_TAG) --network host \
		--build-arg BASE_IMAGE=mlperf-inference-biren:$(DOCKER_TAG)-latest \
		--build-arg GID=$(GROUPID) --build-arg UID=$(UID) --build-arg GROUP=$(GROUPNAME) --build-arg USER=$(UNAME) \
		- < docker/Dockerfile.user
endif

# Add user and launch an interactive container session.
.PHONY: attach_docker
attach_docker:
	@$(MAKE) -f $(MAKEFILE_NAME) docker_add_user
	@$(MAKE) -f $(MAKEFILE_NAME) launch_docker

# Launch a container session.
.PHONY: launch_docker
launch_docker:
ifeq ($(SUPPORT_DOCKER), 1)
	@echo "Launching Docker session"
	$(DOCKER_RUN_CMD) --rm $(DOCKER_INTERACTIVE_FLAGS) -w /work \
		-v $(HOST_VOL):$(CONTAINER_VOL) \
		-v /home/$(UNAME):/home/$(UNAME) \
		--privileged=true \
		-v /dev/biren:/dev/biren:rw \
		-v /home/br104/umd:/home/br104/umd \
		--cap-add SYS_ADMIN --cap-add SYS_TIME \
		--shm-size=32gb \
		-v /etc/timezone:/etc/timezone:ro -v /etc/localtime:/etc/localtime:ro \
		--security-opt apparmor=unconfined --security-opt seccomp=unconfined \
		--name $(BIREN_DOCKER_NAME) -h $(shell echo $(BIREN_DOCKER_NAME) | cut -c -64) --add-host $(BIREN_DOCKER_NAME):127.0.0.1 \
		--user $(UID):$(GROUPID) --net host --device /dev/fuse \
		$(DOCKER_MOUNTS) $(DOCKER_ARGS) \
		-e MLPERF_SCRATCH_PATH=$(MLPERF_SCRATCH_PATH) \
		-e HOST_HOSTNAME=$(HOSTNAME) \
		mlperf-inference-biren:$(DOCKER_TAG) $(DOCKER_COMMAND)
endif

############################## DOWNLOAD_MODEL ##############################

BENCHMARKS ?= resnet50 bert

# Clone LoadGen repo.
.PHONY: clone_loadgen
clone_loadgen:
	@if [ ! -d $(LOADGEN_INCLUDE_DIR) ]; then \
		echo "Cloning Official MLPerf Inference (For Loadgen Files)" \
			&& git clone $(INFERENCE_URL) $(INFERENCE_DIR); \
	fi
	@echo "Updating Loadgen" \
		&& cd $(INFERENCE_DIR) \
		&& git fetch \
		&& git checkout $(INFERENCE_HASH) \
		&& git submodule update --init tools/submission/power-dev \
		&& git submodule update --init third_party/pybind \
		&& git submodule update --init language/bert/DeepLearningExamples \
		&& git submodule update --init vision/medical_imaging/3d-unet-brats19/nnUnet

# Build LoadGen.
.PHONY: build_loadgen
build_loadgen:
	@echo "Building loadgen..."
	@if [ ! -e $(LOADGEN_LIB_DIR) ]; then \
		mkdir $(LOADGEN_LIB_DIR); \
	fi
	@cd $(LOADGEN_LIB_DIR) \
		&& cmake -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) .. \
		&& make -j16
	@if [ ! -e $(LOADGEN_LIB_DIR)/mlperf_loadgen.cpython-38m-$(ARCH)-linux-gnu.so ]; then \
		cd $(LOADGEN_INCLUDE_DIR) \
			&& CFLAGS="-std=c++14 -O3" $(PYTHON3_CMD) setup.py bdist_wheel \
			&& $(PYTHON3_CMD) -m pip install --force-reinstall -t $(LOADGEN_LIB_DIR) dist/mlperf_loadgen-*-linux_$(ARCH).whl; \
	fi

# Build harness source codes.
.PHONY: build_harness
build_harness:
	@echo "Building harness..."
	@mkdir -p build/harness \
		&& cd build/harness \
		&& cmake -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DIS_XAVIER=$(IS_XAVIER) -DLOADGEN_INCLUDE_DIR=$(LOADGEN_INCLUDE_DIR) -DLOADGEN_LIB_DIR=$(LOADGEN_LIB_DIR) -DSUINFER_DIR=$(SUINFER_DIR) -DSUINFER_BUILD_DIR=$(SUINFER_BUILD_DIR) $(PROJECT_ROOT)/code/harness \
		&& make -j
	@echo "Finished building harness."

# Remove build directory.
.PHONY: clean
clean: 
	rm -rf build/bin
	rm -rf build/harness
	rm -rf $(LOADGEN_LIB_DIR)

# Print out useful information.
.PHONY: info
info:
	@echo "RUN_ID=$(RUN_ID)"
	@echo "SYSTEM_NAME=$(SYSTEM_NAME)"
	@echo "Architecture=$(ARCH)"
	@echo "User=$(UNAME)"
	@echo "UID=$(UID)"
	@echo "HOSTNAME=$(HOSTNAME)"
	@echo "Usergroup=$(GROUPNAME)"
	@echo "GroupID=$(GROUPID)"
	@echo "Docker info: {DETACH=$(DOCKER_DETACH), TAG=$(DOCKER_TAG)}"
ifdef DOCKER_IMAGE_NAME
	@echo "Docker image used: $(DOCKER_IMAGE_NAME) -> [$(BASE_IMAGE)]"
endif
	@echo "PYTHON3_CMD=$(PYTHON3_CMD)"
	@echo "PATH=$(PATH)"
	@echo "CPATH=$(CPATH)"
	@echo "LIBRARY_PATH=$(LIBRARY_PATH)"
	@echo "LD_LIBRARY_PATH=$(LD_LIBRARY_PATH)"

# The shell target will start a shell that inherits all the environment
# variables set by this Makefile for convenience.
.PHONY: shell
shell:
	@$(SHELL)
