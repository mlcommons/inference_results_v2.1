import sys
import os
import logging
import time
import array
import json

from pathlib import Path
# import mlperf_loadgen as lg
import inference_utils as infu
from global_vars import *

import numpy as np
import torch
import torch.nn.functional as F
# import torch.autograd.profiler as profiler

import intel_pytorch_extension as ipex
from baseBackend import baseBackend
from unet3d_kits_model import Unet3D

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("3DUnet-Backend")

class Backend(baseBackend):
    def __init__(self, model_path=None, folds=1, checkpoint_name="model_final_checkpoint", verbose=True, calibrate=False, calibration_file="calibration_result.json", *kwargs):
        self.model_path = model_path
        assert Path(model_path).is_file(), "Cannot find the model file {:}!".format(model_path)

        self.verbose = verbose
        self.total_slice_image = 0

        self.int8_conf = None
        if not calibrate:
            if not os.path.isfile(calibration_file):
                log.error("Cannot find int8 calibration file {}".format(calibration_file))
                sys.exit(1)
            self.int8_conf = ipex.AmpConf(torch.int8, calibration_file)


    def load_model(self):
        if self.verbose:
            print("Loading PyTorch model...")
        self.model = torch.jit.load(self.model_path)
        if self.verbose:
            print("Loading PyTorch model...")   
   

    def predict(self, input_data):
        outputs = self.fw_inference(input_data)

        responses = []
        for i in range(len(input_data)):
            responses.append(array.array("B", outputs[i].tobytes()))
        return responses

    def fw_inference(self, data ,calibrate_flag=False, conf=None):
        output = []
        for query in data:
            output.append(self.infer_single_query(query, calibrate_flag, conf))
        return output

    def infer_single_query(self, query, calibrate_flag, conf):
        """
        Performs inference upon data and summarize work with mystr
        Naive implementation of sliding window inference on sub-volume for predetermined
        ROI (Region of Interest) shape is handled here
        Parameters
        ----------
            query: object
                Query sent by LoadGen
        """
        # prepare arrays
        image = query[np.newaxis, ...]
        result, norm_map, norm_patch = infu.prepare_arrays(image, ROI_SHAPE)
        t_image, t_result, t_norm_map, t_norm_patch =\
            self.to_tensor(image), self.to_tensor(result), self.to_tensor(
                norm_map), self.to_tensor(norm_patch)

        # sliding window inference
        subvol_cnt = 0
        for i, j, k in infu.get_slice_for_sliding_window(t_image, ROI_SHAPE, SLIDE_OVERLAP_FACTOR):
            subvol_cnt += 1
            result_slice = t_result[
                ...,
                i:(ROI_SHAPE[0] + i),
                j:(ROI_SHAPE[1] + j),
                k:(ROI_SHAPE[2] + k)]

            input_slice = t_image[
                ...,
                i:(ROI_SHAPE[0] + i),
                j:(ROI_SHAPE[1] + j),
                k:(ROI_SHAPE[2] + k)]

            norm_map_slice = t_norm_map[
                ...,
                i:(ROI_SHAPE[0] + i),
                j:(ROI_SHAPE[1] + j),
                k:(ROI_SHAPE[2] + k)]

            # t1 = time.time()
            result_slice += self.do_infer(input_slice, calibrate_flag, conf) * t_norm_patch
            # t2 = time.time()
            # print("time is ", t2 - t1, "sub_cnt", subvol_cnt)

            norm_map_slice += t_norm_patch

        self.total_slice_image += subvol_cnt
        # print("total slice image ", self.total_slice_image)
        result, norm_map = self.from_tensor(
            t_result), self.from_tensor(t_norm_map)

        final_result = infu.finalize(result, norm_map)
        return final_result

    def do_infer(self, input_tensor, calibrate_flag, conf):
        """
        Perform inference upon input_tensor with PyTorch/TorchScript
        """         
        if calibrate_flag:
            with torch.no_grad():
                with ipex.AutoMixPrecision(conf, running_mode='calibration'):
                    return self.model(input_tensor.to(ipex.DEVICE))
        with torch.no_grad():
            with ipex.AutoMixPrecision(self.int8_conf, running_mode='inference'):
                return self.model(input_tensor.to(ipex.DEVICE))

    def to_tensor(self, my_array):
        """
        Transform numpy array into Torch tensor
        """
        return torch.from_numpy(my_array).float()

    def from_tensor(self, my_tensor):
        """
        Transform Torch tensor into numpy array
        """
        return my_tensor.cpu().numpy().astype(np.float)


    def calibrate(self, samples, qsl, conf_json_file):
        conf = ipex.AmpConf(torch.int8)

        batch_size = len(samples)
        for i in range(batch_size):
            data = []
            data.append(qsl.get_features(samples[i]))
            if self.verbose:
                print("[{:}/{:}] Calibrating sample id {:d} with shape = {:}".format(i, batch_size, samples[i], data[0].shape))
            output = self.fw_inference(data, calibrate_flag=True, conf=conf)
        conf.save(conf_json_file)
        print('calibration_result saved')
        if self.verbose:
            print("Calibration configuration saved to {}".format(conf_json_file))
        self.int8_conf = conf
        

