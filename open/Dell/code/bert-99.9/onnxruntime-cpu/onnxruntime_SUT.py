import array
import os
import sys
import threading
from queue import Queue

import infery
import numpy as np
import onnxruntime as ort

import mlperf_loadgen as lg
from squad_QSL import get_squad_QSL

sys.path.insert(0, os.getcwd())


class BERT_ONNXRuntime_SUT:
    def __init__(self, args):
        self.profile = args.profile
        self.quantized = args.quantized
        self.scenario = args.scenario
        self.batch_size = args.batch_size
        self.count = 0

        # CREATING THE SESSION OPTIONS
        self.model_path = args.model_path
        self.options = ort.SessionOptions()
        self.options.enable_profiling = False
        self.options.log_severity_level = 3
        self.options.intra_op_num_threads = args.intra_threads
        self.options.inter_op_num_threads = args.inter_threads
        self.options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        self.options.execution_mode = (
            ort.ExecutionMode.ORT_SEQUENTIAL
            if args.e_mode == "seq"
            else ort.ExecutionMode.ORT_PARALLEL
        )

        self.inferencer = infery.load(
            model_path=self.model_path,
            framework_type="onnx",
            inference_hardware="cpu",
            session_options=self.options,
            logging_verbosity="ERROR",
        )

        # TODO: ORIGINAL CODE;
        # self.sess = ort.InferenceSession(self.model_path, self.options, providers=['CPUExecutionProvider'])
        print("Constructing SUT...")
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        print("Finished constructing SUT.")
        self.tokenizer_type = args.tokenizer
        self.qsl = get_squad_QSL(args.max_examples, tokenizer_type=args.tokenizer)

        self.workers = []
        self.tasks = Queue(maxsize=args.queue_max_size)
        self.threads = args.worker_threads
        for _ in range(self.threads):
            worker = threading.Thread(target=self.handle_tasks, args=(self.tasks,))
            worker.daemon = True
            self.workers.append(worker)
            worker.start()

    def handle_tasks(self, tasks_queue):
        """Worker thread - reads frames from the task queue and runs inference using this instance's inferencer object."""
        while True:
            qitem = tasks_queue.get()
            if qitem is None:
                # None in the queue indicates the parent want us to exit
                tasks_queue.task_done()
                break

            fd, i = qitem
            res = self.inferencer.predict(**fd)
            self.send_loadgen_response_callback(res, i)
            tasks_queue.task_done()

    def pad_to_batch(self, x):
        x_pad = np.zeros((self.batch_size, x.shape[1]))
        x_pad[: x.shape[0], : x.shape[1]] = x
        return x_pad

    def process_batch(self, batched_features):
        """
        Pre-Processes batch for bert - padded if needed.
        :param batched_features: The features to batch.
        :return: DeciBERT input for ONNXRuntime, padded if needed.
        :rtype: dict
        """
        pad_func = (
            lambda x: self.pad_to_batch(x)
            if len(batched_features) != self.batch_size
            else x
        )

        fd = {
            "input_ids": pad_func(
                np.stack(
                    np.asarray([f.input_ids for f in batched_features]).astype(
                        np.int64
                    )[np.newaxis, :]
                )[0, :, :]
            ),
            "attention_mask": pad_func(
                np.stack(
                    np.asarray([f.input_mask for f in batched_features]).astype(
                        np.int64
                    )[np.newaxis, :]
                )[0, :, :]
            ),
            "token_type_ids": pad_func(
                np.stack(
                    np.asarray([f.segment_ids for f in batched_features]).astype(
                        np.int64
                    )[np.newaxis, :]
                )[
                    0,
                    :,
                ]
            ),
        }

        return fd

    def send_loadgen_response_callback(self, scores, starting_id):
        _scores = [scores[0], scores[1]]
        output = np.stack(_scores, axis=-1)
        for i, sample in enumerate(output):
            response_array = array.array("B", sample.tobytes())
            bi = response_array.buffer_info()
            try:
                response = lg.QuerySampleResponse(
                    self.q_samples[starting_id + i].id, bi[0], bi[1]
                )
                lg.QuerySamplesComplete([response])
            except:
                print(f"starting_id:{starting_id}, len_q_samples:{len(self.q_samples)}")

    def issue_queries(self, query_samples):
        assert self.scenario.lower() == "offline", "only offline scenario is supported"
        eval_features = [
            self.qsl.get_features(query_samples[i].index)
            for i in range(len(query_samples))
        ]
        self.q_samples = query_samples
        stride = self.batch_size
        for i in range(0, len(eval_features), self.batch_size):
            features = eval_features[i : i + stride]
            fd = self.process_batch(features)
            # TODO: UNCOMMENT IF NEEDED FOR DEBUGGING
            # print(f"predicting batch index: {i}")
            self.tasks.put((fd, i))
        print(
            "All the tasks were sent. Waiting for the worker threads to process all of them..."
        )

    def flush_queries(self):
        pass

    def process_latencies(self, latencies_ns):
        pass

    def finish(self):
        print("Waiting for all the threads to finish...")
        for _ in self.workers:
            self.tasks.put(None)
        for worker in self.workers:
            worker.join()

    def __del__(self):
        try:
            self.finish()
            print("Finished destroying SUT.")
        except Exception as e:
            print("ERROOR IN SLEANUP:", e)


def get_onnxruntime_sut(args):
    return BERT_ONNXRuntime_SUT(args)
