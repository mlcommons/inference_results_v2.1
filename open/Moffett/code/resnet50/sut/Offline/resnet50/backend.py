import os
import sys
import ctypes
import importlib
import logging
import collections
import queue
import threading
import time
import tensorflow as tf

import numpy as np
import multiprocessing as mp

sys.path.insert(0, "../../../")
from numpy.ctypeslib import ndpointer
from common.backend import BaseBackend
from common.queue import InputItem, BaseInQueue
from common.utils import wrap_function
from multiprocessing import shared_memory


logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger("MF-MLCommons")
bfloat16 = tf.bfloat16.as_numpy_dtype


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

        self.shm = shared_memory.SharedMemory(create=True, size=32768 * 1024)

    def run(self):
        # TODO: enable affinity feature if performance affected by this
        # os.sched_setaffinity(self.proc_idx, self.affinity)

        # Load model
        log.info(f"WorkerProcess[{self.proc_idx}]: Loading model")
        dataset = importlib.import_module('dataset')

        self.lock.acquire()
        # TODO: add device ordinal
        self.sut_obj = Model(self.model_param, self.system_param, device_ordinal=self.proc_idx)
        self.data_obj = dataset.Dataset(self.model_param, self.dataset_param, self.mlperf_param, self.system_param)

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
                self.shm.close()
                self.shm.unlink()
                self.data_obj.unload_dataset_from_memory()
                break

            query_id_list = next_task.query_id_list
            sample_index_list = next_task.sample_index_list
            output = self.sut_obj.predict(sample_index_list, len(sample_index_list))
            shm_array = np.ndarray(output.shape, dtype=output.dtype, buffer=self.shm.buf)
            shm_array[:] = output[:]
            return_task = (output.shape, output.dtype, query_id_list, self.shm)
            result_queue.put(return_task)
            task_queue.task_done()


class Model:
    # Model parameter
    batch_size = 32768

    def __init__(self, model_param, system_param, device_ordinal=0):
        self.model_path = model_param["model_path"]
        self.library_path = model_param["library_path"]
        self.system_param = system_param
        self.load_dataset_func = None
        self.destroy_func = None
        self.get_output_func = None
        self.output_data_dtype = None
        self.device_ordinal = device_ordinal
        self.model = None
        if not self.system_param['dry_run']:
            self.init()
        log.info(f"WorkerProcess[{self.device_ordinal}]: Dataset loaded")

    def init(self):
        """moffett model initialization.

        """
        model_tester = ctypes.CDLL(self.library_path)
        init_func = wrap_function(model_tester, 'rn50h2h_init', [ctypes.c_char_p, ctypes.c_int], ctypes.c_int)
        ret = init_func(self.model_path.encode('utf-8'), self.device_ordinal)
        if ret != 0:
            raise Exception('Device init error')
        self.model = wrap_function(model_tester, 'rn50h2h_inference',
                                   [ndpointer(ctypes.c_uint32, flags="C_CONTIGUOUS"), ctypes.c_ulong,
                                    ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS")], ctypes.c_int)
        self.destroy_func = wrap_function(model_tester, 'rn50h2h_destroy', None, None)
        self.get_output_func = wrap_function(model_tester, "rn50h2h_get_memory_ptr",
                                        [ctypes.c_ulong, ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS")], ctypes.c_int)
        self.output_data_dtype = "int8"

        log.info(f"WorkerProcess[{self.device_ordinal}]: Model loaded")

    def predict(self, input_index_list, queries_number):
        raw_samples_count = queries_number
        tail_count = queries_number % 128
        if tail_count != 0:
            input_index_list += [0] * (128 - tail_count)
            queries_number += (128 - tail_count)
        input_indices = np.array(input_index_list).astype('uint32')
        output_data = np.zeros([queries_number, 1024, 1], dtype=np.uint8)

        if not self.system_param['dry_run']:
            s = time.time()
            self.model(input_indices, queries_number, output_data)
            e = time.time()
            q = queries_number / (e - s)
        else:
            log.info(f"WorkerProcess[{self.device_ordinal}] Dry run mode will not really execute on device, queries_number:{queries_number}")
        return output_data

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
            consumer = Consumer(
                task_queue=self.input_queues,
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

            if query_samples is None:
                log.info("Receive shutdown sentinel, terminate all subprocesses")
                for i in range(self.process_number):
                    log.info("Put terminator for worker process")
                    self.input_queues.put(None)

                self.input_queues.task_done()

                for c in self.consumers:
                    c.join()
                log.info("All worker processes have been shutdown.")
                return

            self.in_queue.put(query_samples, receipt_time)
