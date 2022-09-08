import logging
import os
import sys

import pyarmnn as ann
from baseBackend import baseBackend

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("BACKEND")


class Backend(baseBackend):
    def __init__(self, model_param, dataset_param):
        self.batch_size = dataset_param["batch_size"]
        self.image_size = dataset_param["image_size"]
        self.precision = dataset_param["precision"]
        self.model_path = model_param["model_path"]
        self.input_layer_name = model_param["input_layer_name"]
        self.output_layer_name = model_param["output_layer_name"]
        self.layout = dataset_param["layout"]
        if not os.path.isfile(self.model_path):
            log.error("Model not found: {}".format(self.model_path))
            sys.exit(1)
        print("Loaded pretrained model")

    def load_model(self):
        print("model_path: " + self.model_path)
        if self.model_path.endswith("tflite"):
            self.parser = ann.ITfLiteParser()
            self.network = self.parser.CreateNetworkFromBinaryFile(self.model_path)
            print(self.network)
            log.info("Model loaded")
            self.input_binding_info = self.parser.GetNetworkInputBindingInfo(0, self.input_layer_name)
            self.output_binding_info = self.parser.GetNetworkOutputBindingInfo(0, self.output_layer_name)
        elif self.model_path.endswith("armnn"):
            self.parser = ann.IDeserializer()
            self.network = self.parser.CreateNetworkFromBinary(self.model_path)
            log.info("Model loaded")
            self.input_binding_info = self.parser.GetNetworkInputBindingInfo(0, self.input_layer_name)
            self.output_binding_info = self.parser.GetNetworkOutputBindingInfo(0, self.output_layer_name)
        else:
            log.error("Unsupported model format!")
            sys.exit(1)
        options = ann.CreationOptions()
        self.runtime = ann.IRuntime(options)
        preferredBackends = [ann.BackendId('CpuAcc')]
        opt_network, messages = ann.Optimize(self.network, preferredBackends, self.runtime.GetDeviceSpec(),
                                             ann.OptimizerOptions(False, False, False,
                                                                  ann.ShapeInferenceMethod_InferAndValidate, False))
        net_id, _ = self.runtime.LoadNetwork(opt_network)
        log.info("Model loaded")

    def predict(self, data):
        input_tensors = ann.make_input_tensors([self.input_binding_info], [data])
        output_tensors = ann.make_output_tensors([self.output_binding_info])
        self.runtime.EnqueueWorkload(0, input_tensors, output_tensors)

        results = ann.workload_tensors_to_ndarray(output_tensors)
        return results[0]
