import ctypes
import os
import logging
import re
import time
from tqdm import tqdm
import numpy as np
import sys

sys.path.insert(0, "../../../")
from common.queue import InputData, OutputItem
from common.utils import wrap_function

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("imagenet")
dataset_array = None


def readlines(filename):
    with open(filename, "r") as f:
        lines = f.read().strip().split("\n")
    return lines


class Dataset:
    def __init__(self, model_param, dataset_param, mlperf_param, system_param):
        self.image_filenames = []
        self.label_list = []
        self.image_list_inmemory = {}
        self.label_list_inmemory = {}
        self.count = mlperf_param["total_sample_count"]
        self.batch_size = dataset_param["batch_size"]
        self.image_size = dataset_param["image_size"]
        self.layout = dataset_param["layout"]
        self.precision = dataset_param["precision"]
        self.dataset_dir = dataset_param["dataset_dir"]
        self.cache_dir = dataset_param["cache_dir"]
        self.cache_path = os.path.join(self.cache_dir, "input.npy")
        self.val_map_path = dataset_param["val_map_path"]
        self.system_param = system_param
        self.preload = True if os.path.exists(self.cache_path) else False
        self.library_path = model_param["library_path"]
        self.model_tester = ctypes.CDLL(self.library_path)

        not_found = 0
        with open(self.val_map_path, 'r') as f:
            for s in f:
                image_name, label = re.split(r"\s+", s.strip())
                src = os.path.join(self.dataset_dir, image_name)
                if not os.path.exists(src):
                    # if the image does not exists ignore it
                    not_found += 1
                    continue
                self.image_filenames.append(src)
                self.label_list.append(int(label))

                # limit the dataset if requested
                if self.count and len(self.image_filenames) >= self.count:
                    break

        self.count = min(self.count, len(self.image_filenames))
        if not system_param['dry_run']:
            self.load_dataset_into_memory()

    def load_query_samples(self, sample_index_list):
        """
        Called by loadgen to load samples before sending queries to sut.
        Ideally complementary to load_dataset. If using this to load samples by loadgen, the samples are not necessarily available across processes - Needs to figure out if possible to work this out
        """
        pass

    def load_dataset_into_memory(self):
        """
        Responsible for loading all available dataset into memory.
        Ideally complementary to 'load_query_samples
        """
        global dataset_array
        log.info(f"Loading dataset into memory, {self.count} images will be loaded")
        load_dataset_func = wrap_function(self.model_tester, 'rn50h2h_load_dataset',
                                          [np.ctypeslib.ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS"),
                                           ctypes.c_ulong],
                                          ctypes.c_int)
        dataset_array = np.array([])
        log.info(f"examples count is {self.count}")
        for index in tqdm(range(self.count)):
            src = self.image_filenames[index]
            if not self.preload:
                processed_array = self.pre_process(src)
                if dataset_array.any():
                    dataset_array = np.concatenate([dataset_array, np.array([processed_array])], axis=0)
                else:
                    dataset_array = np.array([processed_array])
            self.image_list_inmemory[index] = src
            self.label_list_inmemory[index] = self.label_list[index]

        if self.preload:
            log.info(f"Preload Datasets ......")
            dataset_array = np.load(self.cache_path)
        else:
            log.info(f"Preprocessing the datasets for inference")
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)
            if dataset_array.any():
                np.save(self.cache_path, dataset_array)
        log.info(f"input dataset_array shape is {dataset_array.shape}")
        load_dataset_func(dataset_array, dataset_array.nbytes)

    def unload_dataset_from_memory(self):
        unload_func = wrap_function(self.model_tester, 'rn50h2h_destroy', None, None)
        unload_func()

    def load_dataset(self):
        """
        Responsible for loading all available dataset into memory.
        Ideally complementary to 'load_query_samples
        """

    def unload_query_samples(self, sample_list):
        """
        Workload dependent. But typically not implemented if load_query_samples is not implemented
        """
        log.info("Called to unload data")
        pass

    def obj_unload_query_samples(self, sample_list):
        if sample_list:
            for sample in sample_list:
                if sample in self.image_list_inmemory:
                    del self.image_list_inmemory[sample]
                    del self.label_list_inmemory[sample]
        else:
            self.image_list_inmemory = {}
            self.label_list_inmemory = {}

    def get_samples(self, sample_index_list):
        """
        Fetches and returns pre-processed data at requested 'sample_index_list'
        """
        return InputData(data=sample_index_list, data_shape=len(sample_index_list))

    def get_warmup_samples(self):
        """
        Fetches and returns pre-processed data for warmup
        """
        import random
        num_samples = self.batch_size
        warmup_samples = []
        outData = []
        if len(self.image_list_inmemory) < num_samples:
            self.load_query_samples(list(range(num_samples)))
        sample_ids = random.choices(list(self.image_list_inmemory.keys()), k=num_samples)
        item = InputData(data=sample_ids, data_shape=len(sample_ids))
        warmup_samples.append(item)
        return warmup_samples

    def pre_process(self, file=""):
        """
        Pre-processes a given input/image
        """
        import cv2
        import numpy as np
        import tensorflow as tf

        img = cv2.imread(file)

        ## convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ## resize image with short edge as 288
        interpolation = cv2.INTER_LINEAR
        short_edge = 256
        img_h, img_w, _ = img.shape
        scale = short_edge * 1.0 / min(img_h, img_w)
        new_w = int(np.ceil(img_w * scale))
        new_h = int(np.ceil(img_h * scale))
        if new_h != img_h or new_w != img_w:
            img = cv2.resize(img, (new_w, new_h), interpolation=interpolation)

        # center crop 256
        target_size = 224
        sy = (new_h - target_size) // 2
        sx = (new_w - target_size) // 2
        img = img[sy:sy + target_size, sx:sx + target_size]

        # normalize
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img = (np.float32(img) / 255. - mean) / std

        # # transpose HWC -> CHW, 模型输入为NHWC, 所以不用transpose
        # img = img.transpose(2, 0, 1)

        # quantize input
        output_max = 2.6238868230138905
        scale = 127 / output_max
        img = tf.cast(
            tf.round(
                tf.cast(
                    tf.clip_by_value(
                        tf.cast(img, tf.bfloat16) * tf.cast(scale, tf.bfloat16), -128, 127
                    ),
                    tf.float32)
            ),
            tf.int8
        )
        img = img.numpy()
        return np.frombuffer(img.flatten().tobytes(order='C'), dtype="uint8")

    def post_process(self, query_ids, sample_index_list, results):
        """
        Post-processor that accepts loadgens query ids and corresponding inference output.
        post_process should return and OutputItem object which has two attributes:
        OutputItem.query_id_list
        OutputItem.results
        """
        processed_results = []
        results = np.argmax(results, axis=1)
        n = results.shape[0]
        for idx in range(n):
            result = results[idx]
            processed_results.append([result])
        ret = OutputItem(query_ids, processed_results, array_type_code='q')
        ret.receipt_time = time.time()
        return ret
