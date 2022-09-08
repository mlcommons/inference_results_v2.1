import os
import sys
import ctypes
import importlib
import logging
import collections
import queue
import threading
import time
import multiprocessing as mp
import numpy as np
import tensorflow as tf
from multiprocessing import shared_memory
from numpy.ctypeslib import ndpointer

sys.path.insert(0, "../../../")
from common.backend import BaseBackend
from common.queue import InputItem, BaseInQueue
from helper import sentence_preprocess

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("MF-MLCommons")
bfloat16 = tf.bfloat16.as_numpy_dtype


def wrap_function(lib, func_name, argtypes, restype):
    """Simplify wrapping ctypes functions"""
    func = lib.__getattr__(func_name)
    func.argtypes = argtypes
    func.restype = restype
    return func


class InQueue(BaseInQueue):
    def __init__(self, mpQueue=None, batch_size=1, **kwargs):
        """
        Will be instantiated with keyword-val dictionary
        Initializer should have named inputs.
        As a generic rull, init should have named input 'mpQueue' which is a multiprocessing module's JoinableQueue instance
        """
        # TODO: We may chose to pass mpQueue as list even if not bucketing
        super().__init__(mpQueue, batch_size, **kwargs)
        self.in_queue = mpQueue if isinstance(mpQueue, list) else [mpQueue]
        self.batch_size = batch_size
        self.curr_query_count = 0
        self.qid_list = []
        self.qidx_list = []

    def put(self, query_samples, receipt_time=0):
        """
        Receives query sample(s) from loadgen and processes the queries in batches if required/desired.
        Processed/batched queries are wrapped in 'InputItem' object, and then placed on the producer queue
        """
        num_samples = len(query_samples)
        # TODO: Remove all logging in here
        # log.debug("Adding {} samples to queue".format(num_samples))
        if num_samples == 1:
            self.curr_query_count += 1
            self.qid_list.append(query_samples[0].id)
            self.qidx_list.append(query_samples[0].index)
            if self.curr_query_count == self.batch_size:
                item = InputItem(self.qid_list, self.qidx_list, receipt_time=receipt_time)
                w_idx = np.random.randint(0, len(self.in_queue))
                self.in_queue[w_idx].put(item)
                self.curr_query_count = 0
                self.qid_list = []
                self.qidx_list = []

        else:
            idx = [q.index for q in query_samples]
            query_id = [q.id for q in query_samples]

            num_batches = num_samples // self.batch_size
            remainder = num_samples % self.batch_size
            batch = 0
            bidx = 0
            bs = self.batch_size
            while batch < num_batches:
                ids = query_id[bidx:bidx + bs]
                indexes = idx[bidx:bidx + bs]
                item = InputItem(ids, indexes, receipt_time=receipt_time)  # , data, label)

                w_idx = np.random.randint(0, len(self.in_queue))
                self.in_queue[w_idx].put(item)
                batch += 1
                bidx += bs

            if remainder > 0:
                ids = query_id[bidx:]
                indexes = idx[bidx:]
                item = InputItem(ids, indexes)  # , data, label)
                self.in_queue[0].put(item)

    def put_last_batch(self, receipt_time=0):
        """
        Receives query sample(s) from loadgen and processes the queries in batches if required/desired.
        Processed/batched queries are wrapped in 'InputItem' object, and then placed on the producer queue
        """
        if self.qid_list:
            item = InputItem(self.qid_list, self.qidx_list, receipt_time=receipt_time)
            self.in_queue[0].put(item)


class Consumer(mp.Process):
    def __init__(self, task_queue, out_queue, lock, init_counter, proc_idx, model_cls, model_param, system_param, dataset_param,
                 mlperf_param):
        super(Consumer, self).__init__()

        self.sut_obj = None
        self.data_obj = None
        self.task_queue = task_queue
        self.out_queue = out_queue
        self.lock = lock
        self.init_counter = init_counter
        self.proc_idx = proc_idx
        # self.num_cores = system_param["core_per_instance"]
        self.workers = []
        self.warmup_count = 0  # TODO: do we need warmup?
        self.latencies = collections.defaultdict(list)
        # self.num_workers = system_param["num_worker_per_instance"]
        # self.core_per_worker = system_param["core_per_worker"]
        self.model_param = model_param
        self.dataset_param = dataset_param
        self.system_param = system_param
        self.model_cls = model_cls
        self.mlperf_param = mlperf_param
        self.shm_0 = shared_memory.SharedMemory(create=True, size=10833 * 4 * 384 * 2 * 4)
        self.shm_1 = shared_memory.SharedMemory(create=True, size=10833 * 4 * 384 * 2 * 4)
        self.shm_2 = shared_memory.SharedMemory(create=True, size=10833 * 4 * 4096)

    def run(self):
        # TODO: enable affinity feature if performance affected by this
        # os.sched_setaffinity(self.proc_idx, self.affinity)

        # Load model
        log.info(f"WorkerProcess[{self.proc_idx}]: Loading model")
        # from .dataset import Dataset
        dataset = importlib.import_module('dataset')

        self.lock.acquire()
        # TODO: add device ordinal
        self.sut_obj = Model(self.model_param, self.dataset_param, self.system_param, device_ordinal=self.proc_idx)
        self.data_obj = dataset.Dataset(self.dataset_param, self.mlperf_param)

        self.data_obj.load_dataset()
        log.info(f"WorkerProcess[{self.proc_idx}]:Available samples: {self.data_obj.count}")

        self.lock.release()

        if self.warmup_count > 0:
            self.do_warmup()

        self.handle_tasks(0, self.task_queue, self.out_queue, self.pid)

    def do_warmup(self):
        pass

    def handle_tasks(self, i, task_queue, result_queue, pid):
        pid = os.getpid()

        worker_name = str(pid) + "-" + str(i)

        self.lock.acquire()
        self.init_counter.value += 1
        self.lock.release()

        # TODO: Enable profiling here
        while True:
            next_task = task_queue.get()
            if next_task is None:
                log.info("{} : Exiting ".format(worker_name))
                self.sut_obj.destroy_func()
                self.shm_0.close()
                self.shm_0.unlink()
                self.shm_1.close()
                self.shm_1.unlink()
                self.shm_2.close()
                self.shm_2.unlink()
                break
            #
            # query_id_list = next_task.query_id_list
            # sample_index_list = next_task.sample_index_list
            query_id_list, query_examples, padding_output, padding_output_mask, preprocess_mapping_indices = next_task

            output_data, queries_number_before_padding = self.sut_obj.predict(query_examples)

            shm_array_0 = np.ndarray(shape=padding_output.shape, dtype=padding_output.dtype, buffer=self.shm_0.buf)
            shm_array_0[:] = padding_output[:]

            shm_array_1 = np.ndarray(shape=padding_output_mask.shape, dtype=padding_output_mask.dtype,
                                     buffer=self.shm_1.buf)
            shm_array_1[:] = padding_output_mask[:]

            shm_array_2 = np.ndarray(shape=output_data.shape, dtype=output_data.dtype, buffer=self.shm_2.buf)
            shm_array_2[:] = output_data[:]

            return_task = (query_id_list,
                           self.shm_0, padding_output.shape, padding_output.dtype,
                           self.shm_1, padding_output_mask.shape, padding_output_mask.dtype,
                           self.shm_2, output_data.shape, output_data.dtype,
                           preprocess_mapping_indices, queries_number_before_padding)

            result_queue.put(return_task)
            task_queue.task_done()

class Model:
    # Model parameter
    batch_size = 10833

    def __init__(self, model_param, dataset_param, system_param, device_ordinal=0):
        self.library_path = model_param["library_path"]
        self.model_path = model_param["model_path"]
        self.system_param = system_param
        self.batch_size = dataset_param["batch_size"]
        self.max_seq_length = dataset_param["max_seq_length"]
        self.doc_stride = dataset_param["doc_stride"]
        self.max_query_length = dataset_param["max_query_length"]
        self.dataset_path = dataset_param["dataset_path"]
        self.vocab_path = dataset_param["vocab_path"]
        self.cache_dir = dataset_param["cache_dir"]
        self.device_ordinal = device_ordinal
        self.model = None
        self.load_dataset_func = None
        self.destroy_func = None
        self.output_data_dtype = None
        self.dataset_array = None

        # self.shm = None
        if not self.system_param['dry_run']:
            self.init()

    def init(self):
        """moffett model initialization.

        """
        model_tester = ctypes.CDLL(self.library_path)
        flag_id = 0
        init_func = wrap_function(model_tester, 'bertlh2h_init', [ctypes.c_char_p, ctypes.c_int, ctypes.c_int],
                                  ctypes.c_int)
        ret = init_func(self.model_path.encode('utf-8'), self.device_ordinal, flag_id)
        if ret != 0:
            raise Exception('Device init error')

        self.model = wrap_function(model_tester, 'bertlh2h_inference',
                                   [ndpointer(ctypes.c_uint32, flags="C_CONTIGUOUS"), ctypes.c_ulong,
                                    ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS")], ctypes.c_int)
        self.load_dataset_func = wrap_function(model_tester, "bertlh2h_load_dataset",
                                               [ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS"), ctypes.c_ulong,
                                                ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS"), ctypes.c_ulong,
                                                ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS"), ctypes.c_ulong],
                                               ctypes.c_int)
        self.destroy_func = wrap_function(model_tester, 'bertlh2h_destroy', None, None)
        self.output_data_dtype = bfloat16
        log.info("Model loaded")

    def predict(self, query_examples):
        preprocess_input_ids = query_examples[0]
        preprocess_segment_ids = query_examples[1]
        preprocess_input_mask = query_examples[2]
        preprocess_samples_count = preprocess_input_ids.shape[0]
        if not self.system_param['dry_run']:
            self.load_dataset_func(preprocess_input_ids, preprocess_input_ids.nbytes,
                                   preprocess_segment_ids, preprocess_segment_ids.nbytes,
                                   preprocess_input_mask, preprocess_input_mask.nbytes)
            # log.info("Loading dataset Success")
        else:
            pass
        # after preprocess
        input_index_list = list(range(0, preprocess_samples_count))
        queries_number_before_padding = preprocess_samples_count
        queries_number = queries_number_before_padding

        # must be multiplier of 128
        tail_count = queries_number % 128
        if tail_count != 0:
            # log.info(f"WorkerProcess[{self.device_ordinal}]: queries_number before padding: {queries_number}")
            input_index_list += [0] * (128 - tail_count)
            queries_number += (128 - tail_count)
            # log.info(f"WorkerProcess[{self.device_ordinal}]: queries_number before padding: {queries_number}")

        input_indices = np.array(input_index_list).astype('uint32')
        output_data = np.zeros([queries_number, 4096], dtype=np.uint8)


        if not self.system_param['dry_run']:
            self.model(input_indices, queries_number, output_data)
        else:
            log.info(f"WorkerProcess[{self.device_ordinal}]: "
                     f"Dry run mode will not really execute on device, queries_number: {queries_number}")
        return output_data, queries_number_before_padding

class PreprocessWorker(mp.Process):
    def __init__(self, in_queue: mp.JoinableQueue, out_queue: mp.JoinableQueue, init_value: mp.Value, init_value_lock: mp.Lock, device_ordinal, cache_dir,
                 vocab_path, dataset_path, max_seq_length, doc_stride, max_query_length, **kwargs):
        super().__init__(**kwargs)
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.daemon = True
        self.init_value = init_value
        self.init_value_lock = init_value_lock
        self.dataset_array = None
        self.cache_dir = cache_dir  # TODO: pass cache_dir into PreprocessWorker

        # Load a copy of dataset in PreprocessWorker process
        self.load_dataset()
        self.device_ordinal = device_ordinal
        self.cache_dir = cache_dir
        self.vocab_path = vocab_path
        self.dataset_path = dataset_path
        self.max_seq_length = max_seq_length
        self.doc_stride = doc_stride
        self.max_query_length = max_query_length


    def handle_item(self, item):
        query_id_list = item.query_id_list
        sample_index_list = item.sample_index_list

        # TODO: self.data_array should be shm
        input_data = self.dataset_array[sample_index_list]
        query_examples, padding_output, padding_output_mask, preprocess_mapping_indices = sentence_preprocess(input_data)

        return query_id_list, query_examples, padding_output, padding_output_mask, preprocess_mapping_indices

    def load_dataset(self):
        log.info("Loading dataset")
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        input_names = ["input_ids", "segment_ids", "input_mask"]
        cache_path_list = [os.path.join(self.cache_dir, f"{name}.npy") for name in input_names]
        preload = True
        for file_path in cache_path_list:
            if not os.path.exists(file_path):
                preload = False
        if preload:
            dataset_samples = [np.load(file_path) for file_path in cache_path_list]
        else:
            from transformers import BertTokenizer
            from create_squad_data import read_squad_examples, convert_examples_to_features
            tokenizer = BertTokenizer(self.vocab_path)
            eval_examples = read_squad_examples(input_file=self.dataset_path,
                                                is_training=False, version_2_with_negative=False)

            def append_feature(feature):
                eval_features.append(feature)

            eval_features = []
            convert_examples_to_features(
                examples=eval_examples,
                tokenizer=tokenizer,
                max_seq_length=self.max_seq_length,
                doc_stride=self.doc_stride,
                max_query_length=self.max_query_length,
                is_training=False,
                output_fn=append_feature,
                verbose_logging=False)

            dataset_samples = []
            for name in input_names:
                input_array = np.array([getattr(feature, name) for feature in eval_features])
                cache_path = os.path.join(self.cache_dir, f"{name}.npy")
                np.save(cache_path, input_array)
                dataset_samples.append(input_array)

        self.dataset_array = np.array(dataset_samples)
        self.dataset_array = np.transpose(np.array(self.dataset_array), (1, 2, 0))

    def run(self) -> None:
        self.init_value_lock.acquire()
        self.init_value.value += 1
        self.init_value_lock.release()
        while True:
            item = self.in_queue.get()
            # log.info(f'PreprocessWorker got input')
            if item is None:
                self.out_queue.put(None)
                break

            result = self.handle_item(item)

            # log.info(f'PreprocessWorker put output')
            self.out_queue.put(result)

class Backend(BaseBackend):
    def __init__(self, config, query_queue: queue.Queue, out_queue, initialized_event: threading.Event):
        super().__init__(config, query_queue, out_queue)
        self.config = config
        self.query_queue = query_queue  # Thread-safe queue
        self.out_queue = out_queue  # Process-safe Queue
        self.consumers = []
        self._worker_initialized = False
        if config["system_param"]["platform"] == 's4':
            self.process_number = 1
        elif config["system_param"]["platform"] == 's10':
            self.process_number = 2
        else:
            self.process_number = 3

        self.input_queues = mp.JoinableQueue()
        self.in_queue = InQueue(self.input_queues, Model.batch_size)

        self._initialize_worker(initialized_event)

    def _initialize_worker(self, initialized_event: threading.Event):
        if self._worker_initialized:
            raise RuntimeError('Re-initialize worker subprocesses, exit.')

        # Establish communication queues
        self.lock = mp.Lock()
        init_counter = mp.Value("i", 0)

        # spawn 1 or 3 processes for tests

        for i in range(self.process_number):
            preprocess_output_queue = mp.JoinableQueue()
            preprocess_worker = PreprocessWorker(
                in_queue=self.input_queues,
                out_queue=preprocess_output_queue,
                init_value=init_counter,
                init_value_lock=self.lock,
                device_ordinal=i,
                cache_dir=self.config["dataset_param"]["cache_dir"],
                vocab_path=self.config["dataset_param"]["vocab_path"],
                dataset_path=self.config["dataset_param"]["dataset_path"],
                max_seq_length=self.config["dataset_param"]["max_seq_length"],
                doc_stride=self.config["dataset_param"]["doc_stride"],
                max_query_length=self.config["dataset_param"]["max_query_length"]
            )

            self.consumers.append(preprocess_worker)

            consumer = Consumer(
                task_queue=preprocess_output_queue,
                out_queue=self.out_queue,
                lock=self.lock,
                init_counter=init_counter,
                proc_idx=i,
                model_cls=Model,
                model_param=self.config["model_param"],
                system_param=self.config["system_param"],
                dataset_param=self.config["dataset_param"],  # TODO: pass dataset parameter from args
                mlperf_param=self.config["mlperf_param"]
            )

            self.consumers.append(consumer)

        self._worker_initialized = True

        for consumer in self.consumers:
            # Start the subprocess to serve model
            consumer.start()

        while init_counter.value < self.process_number:
            time.sleep(2)
            log.info("Wait for all subprocesses ready")

        # Notify main thread continuing
        initialized_event.set()

    def run(self):
        # Get query from loadgen
        while True:
            # For offline mode, all queries will be packed into one request
            query_samples, receipt_time = self.query_queue.get()

            # receive shutdown sentinel for process_number times
            if query_samples is None:
                log.info("Receive shutdown sentinel, terminate all subprocesses")
                for i in range(self.process_number):
                    self.input_queues.put(None)

                for c in self.consumers:
                    c.join()

                return

            self.in_queue.put(query_samples, receipt_time)
