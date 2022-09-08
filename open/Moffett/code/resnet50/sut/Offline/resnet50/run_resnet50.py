import multiprocessing.shared_memory
import sys
import multiprocessing as mp
import collections
import time
import yaml
import os
import queue
import threading
import logging
from argparse import ArgumentParser
import array
import mlperf_loadgen as lg
import numpy as np
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger("MF-MLCommons")

SCENARIO_MAP = {
    # "multistream": lg.TestScenario.MultiStream,
    # "singlestream": lg.TestScenario.SingleStream,
    "offline": lg.TestScenario.Offline,
    "server": lg.TestScenario.Server,
}

RESOURCES = [
    'mlperf_resnet50_offline',
    'mlperf_resnet50_server',
    'mlperf_bert_offline',
    'mlperf_bert_server'
]


def get_args():
    parser = ArgumentParser("Parses global and workload-specific arguments")
    parser.add_argument("--config", required=True)
    parser.add_argument('--mode', required=True,
                        choices=('AccuracyOnly', 'PerformanceOnly', 'SubmissionRun', 'FindPeakPerformance'))
    parser.add_argument('--output_dir', required=False, default="output_logs")
    args = parser.parse_args()
    return args


def enumerate_spu():
    pass


# input_shm = multiprocessing.shared_memory.SharedMemory(create=True, size=queries_number * 1024)
def main():
    args = get_args()

    with open(args.config, encoding='utf8') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    settings = lg.TestSettings()
    settings.scenario = SCENARIO_MAP[config["mlperf_param"]["scenario"].lower()]
    settings.FromConfig(config["mlperf_param"]["mlperf_conf"], config["mlperf_param"]["workload"],
                        config["mlperf_param"]["scenario"])
    settings.FromConfig(config["mlperf_param"]["user_conf"], config["mlperf_param"]["workload"],
                        config["mlperf_param"]["scenario"])
    settings.mode = lg.TestMode.AccuracyOnly if args.mode.lower() == "accuracyonly" else lg.TestMode.PerformanceOnly

    # ============== Output Collector ==============
    out_queue = mp.Queue()

    # Start response thread
    resp_worker = threading.Thread(
        target=response_loadgen, args=(out_queue,))

    resp_worker.daemon = True
    resp_worker.start()

    # ============== Input Queue ==============
    in_queue = queue.Queue()

    def issue_queries(query_samples):
        in_queue.put((query_samples, time.time()))  # (query_samples: List, receipt_time: float)

    # ============== Start SUT Thread ==============
    sut_initialized = threading.Event()
    sut_worker = threading.Thread(
        target=sut_handler, args=(config, in_queue, out_queue, sut_initialized))
    sut_worker.daemon = True
    sut_worker.start()

    # Wait SUT initialized
    log.info("Waiting SUT initialized")
    sut_initialized.wait()
    log.info("SUT initialized")

    # ============== Config Loadgen ==============
    sut = lg.ConstructSUT(issue_queries, flush_queries)
    qsl = lg.ConstructQSL(
        config["mlperf_param"]['total_sample_count'],
        min(config["mlperf_param"]['total_sample_count'], settings.performance_sample_count_override),
        load_query_samples,
        unload_query_samples)

    log_path = args.output_dir
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = log_path
    log_output_settings.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings

    # ============== Start Loadgen ==============
    lg.StartTestWithLogSettings(sut, qsl, settings, log_settings)
    log.info("Test completed")

    # send sentinel to kill SUT thread
    in_queue.put((None, time.time()))

    # Kill response collector thread
    out_queue.put(None)

    log.info("Will destroy SQL and SUT")
    lg.DestroyQSL(qsl)
    lg.DestroySUT(sut)
    log.info("SQL and SUT shutdown")


def sut_handler(config, in_queue, out_queue, finished: threading.Event):
    """
    SUT thread handler
    @param args: arguments of program
    @param settings: mlcommons loadgen setting object
    @param in_queue: feeding queue
    @param out_queue: fetching queue
    @param finished: Event for the end of initialization
    """

    # Add specified model for PYTHONPATH
    sys.path.insert(0, os.path.join(os.getcwd(), 'sut', config["mlperf_param"]['scenario'],
                                    config["mlperf_param"]['workload']))

    import importlib
    # backend = importlib.import_module(
    #     f'sut.{config["mlperf_param"]["scenario"]}.{config["mlperf_param"]["workload"]}.backend')
    backend = importlib.import_module('backend')

    # method run() should be implemented as busy loop
    sut = backend.Backend(config, in_queue, out_queue, finished)

    # Start loop
    sut.run()


def flush_queries():
    pass


def process_latencies(latencies):
    pass


def load_query_samples(query_samples):
    pass


def unload_query_samples(query_samples):
    pass


def response_loadgen(out_queue):
    def post_process(output_data):
        """
        Post-processor that accepts loadgens query ids and corresponding inference output.
        post_process should return and OutputItem object which has two attributes:
        OutputItem.query_id_list
        OutputItem.results
        """
        results = np.frombuffer(output_data.flatten().tobytes(order='C'), dtype="int8").reshape(
            [output_shape[0], -1])[:query_number]
        return np.argmax(results, axis=1)

    while True:
        next_task = out_queue.get()

        # Return if we receive shutdown sentinel
        if next_task is None:
            # None means shutdown
            log.info('Exiting response thread')
            break

        output_shape, output_dtype, query_id_list, shm = next_task

        # log.info(f'shm key: {shm.name}')

        next_task_shm = np.ndarray(output_shape, dtype=output_dtype, buffer=shm.buf)
        query_number = len(query_id_list)
        result = post_process(next_task_shm)
        array_type_code = 'q'
        for i in range(query_number):
            response_array = array.array(array_type_code, [result[i]])
            bi = response_array.buffer_info()
            responses = [lg.QuerySampleResponse(query_id_list[i], bi[0], bi[1] * response_array.itemsize)]
            lg.QuerySamplesComplete(responses)

        shm.close()
        # shm.unlink()


if __name__ == '__main__':
    mp.set_start_method("spawn")

    main()
