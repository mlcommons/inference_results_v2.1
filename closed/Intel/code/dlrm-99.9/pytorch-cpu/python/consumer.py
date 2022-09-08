from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import array
import numpy as np
import os
import multiprocessing
import time
import torch
import torch.multiprocessing as mp
import intel_pytorch_extension as ipex
from items import OItem

class Consumer(multiprocessing.Process):
    def __init__(self, task_queue, result_queue, ds_queue, lock, init_counter, proc_num, args, first_instance_start_core, cpus_per_socket, cpus_per_process, cpus_per_instance):
        multiprocessing.Process.__init__(self)
        self.args = args
        self.device = ipex.DEVICE
        self.ipex_conf = None
        if args.use_int8:
            self.ipex_conf = ipex.AmpConf(torch.int8, args.int8_configuration_dir)
        self.lock = lock
        self.ds_queue = ds_queue
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.rqnum = len(result_queue)
        self.init_counter = init_counter
        self.cpus_per_process = cpus_per_process
        self.proc_num = proc_num
        self.workers = []
        self.instances_start_core = []
        self.instances_end_core = []
        #self.instances_affinity = []
        self.instances_core_nums = []
        self.instances_core_id = []

        procs_per_socket = cpus_per_socket // cpus_per_process
        socket_num = self.proc_num // procs_per_socket
        socket_proc_idx = self.proc_num % procs_per_socket
        self.start_core_idx = socket_num * cpus_per_socket + socket_proc_idx * self.cpus_per_process
        self.end_core_idx = self.start_core_idx + self.cpus_per_process
        if self.proc_num == 0:
            self.start_core_idx = self.start_core_idx + first_instance_start_core  #reserved threads for loadgen staff
        self.affinity = range(self.start_core_idx, self.end_core_idx)
        self.core_nums = self.end_core_idx - self.start_core_idx
        self.num_ins = self.core_nums // cpus_per_instance
        if self.proc_num == 0 and (first_instance_start_core > 0):
            left_cores = self.core_nums - self.num_ins * cpus_per_instance
            if left_cores >= (cpus_per_instance // 2):
                self.num_ins = self.num_ins + 1

        for i in range(self.num_ins):
            if self.proc_num == 0:
              if (cpus_per_instance - first_instance_start_core) >= (cpus_per_instance // 2):
                  if i == 0:
                      self.instances_start_core.append(first_instance_start_core)
                      self.instances_core_id.append(first_instance_start_core)
                      self.instances_end_core.append(cpus_per_instance)
                  else:
                      self.instances_start_core.append(i * cpus_per_instance)
                      self.instances_end_core.append(self.instances_start_core[i] + cpus_per_instance)
                      self.instances_core_id.append(i * cpus_per_instance)
              else:
                  self.instances_start_core.append((i + 1) * cpus_per_instance)
                  self.instances_end_core.append(self.instances_start_core[i] + cpus_per_instance)
                  self.instances_core_id.append((i + 1) * cpus_per_instance)
            else:
                  self.instances_start_core.append(self.start_core_idx + i * cpus_per_instance)
                  self.instances_end_core.append(self.instances_start_core[i] + cpus_per_instance)
                  self.instances_core_id.append(i * cpus_per_instance)
            #self.instances_affinity.append(range(self.instances_start_core[i], self.instances_end_core[i]))
            self.instances_core_nums.append(self.instances_end_core[i] - self.instances_start_core[i])

    def input_wrap(self, X, lS_o, lS_i, use_gpu):
        if self.args.use_gpu or self.args.use_ipex:
           lS_i = [S_i.to(self.device) for S_i in lS_i] if isinstance(lS_i, list) else lS_i.to(self.device)
           lS_o = [S_o.to(self.device) for S_o in lS_o] if isinstance(lS_o, list) else lS_o.to(self.device)
           X = X.to(self.device)
           return X, lS_o, lS_i

    def model_predict(self, batch_dense_X, batch_lS_o, batch_lS_i):
        X, lS_o, lS_i = self.input_wrap(batch_dense_X, batch_lS_o, batch_lS_i, self.args.use_gpu)
        if self.args.use_int8:
            with torch.no_grad():
                with ipex.AutoMixPrecision(self.ipex_conf, running_mode='inference'):
                    output = self.model(X, lS_o, lS_i)
        else:
            with torch.no_grad():
                output = self.model(X, lS_o, lS_i)
        return output

    def warmup(self, model):
        torch.set_grad_enabled(False)
        ipex.core.enable_auto_dnnl()
        ipex.core.set_execution_mode(False)
        self.lS_o_buffer = {}
        for s in range(self.args.max_batchsize, self.args.max_batchsize + 800, 100):
            batch_dense_X = torch.randn((s, 13), dtype=torch.float)
            batch_lS_i = torch.ones([26, s], dtype=torch.long)
            batch_lS_o = torch.LongTensor(26, s)
            for i in range(26):
                batch_lS_o[i] = torch.arange(s)
            self.lS_o_buffer[s] = batch_lS_o
            self.model_predict(batch_dense_X, batch_lS_o, batch_lS_i)

    # def trace(self, ds, model):
    #     batch_dense_X = torch.randn((self.args.max_batchsize, 13), dtype=torch.float)
    #     batch_lS_i = torch.ones([26, self.args.max_batchsize], dtype=torch.long)
    #     batch_lS_o = torch.stack([torch.arange(self.args.max_batchsize) for _ in range(26)])
    #     model.trace(batch_dense_X, batch_lS_o, batch_lS_i)

    def get_samples(self, id_list):
        ls = []
        for i in id_list:
            ls.append(self.items_in_memory[i])
        ls_t = list(zip(*ls))

        #X = ipex.core.concat_all_continue(ls_t[0], 0)
        X = torch.cat(ls_t[0])
        lS_i = ipex.core.concat_all_continue(ls_t[1], 1)
        T = ipex.core.concat_all_continue(ls_t[2], 0)
        (num_s, len_ls) = lS_i.size()
        if len_ls in self.lS_o_buffer:
            return (X, self.lS_o_buffer[len_ls], lS_i, T)
        else:
            lS_o = torch.LongTensor(num_s, len_ls)
            for i in range(num_s):
                lS_o[i] = torch.arange(len_ls)
            return (X, lS_o, lS_i, T)

    def handle_tasks(self, i, task_queue, result_queue, args, pid):
        #os.sched_setaffinity(self.workers[i].pid, self.instances_affinity[i])
        ipex.core.thread_bind(self.proc_num, self.cpus_per_process, self.instances_core_id[i], self.instances_core_nums[i])

        instance_name = str(pid) + "-" + str(i)
        #print(instance_name, " : Start handle_tasks")
        if args.enable_profiling:
            filename = "dlrm_mlperf_offline_run_" + instance_name + ".prof"

        self.lock.acquire()
        self.init_counter.value += 1
        self.lock.release()

        with torch.autograd.profiler.profile(args.enable_profiling) as prof:
          while True:
            qitem = task_queue.get()
            if qitem is None:
                if args.enable_profiling:
                    with open(filename, "w") as prof_f:
                        prof_f.write(prof.key_averages().table(sort_by="self_cpu_time_total"))
                #print(instance_name, " : Exit")
                break

            #get_sample_start = time.time()
            batch_dense_X, batch_lS_o, batch_lS_i, batch_T = self.get_samples(qitem.content_id)
            idx_offsets = qitem.idx_offsets
            #get_sample_timing = time.time() - get_sample_start
            #print("DS get_samples elapsed time:{} ms ".format(get_sample_timing * 1000))
            presults = []
            try:
                #predict_start = time.time()
                results = self.model_predict(batch_dense_X, batch_lS_o, batch_lS_i)
                #predict_timing = time.time() - predict_start
                #print("batch size = {}, predict elapsed time:{} ms".format(len(batch_dense_X), predict_timing * 1000))
                # post_process
                results = results.detach().cpu()
                presults = ipex.core.concat_all_continue((results, batch_T), 1)
                # presults = torch.cat((results, batch_T), dim=1)

                if args.accuracy:
                    total = len(results)
                    good = (results.round() == batch_T).nonzero(as_tuple=False).size(0)
                    result_timing = time.time() - qitem.start

            except Exception as ex:  # pylint: disable=broad-except
                log.error("instance ", instance_name, " failed, %s", ex)
                presults = [[]] * len(qitem.query_id)
            finally:
                response_array_refs = []
                query_list = qitem.query_id
                prev_off = 0
                for idx, query_id in enumerate(query_list):
                    cur_off = prev_off + idx_offsets[idx]
                    response_array = array.array("B", np.array(presults[prev_off:cur_off], np.float32).tobytes())
                    response_array_refs.append(response_array)
                    prev_off = cur_off
                if args.accuracy:
                    result_queue.put(OItem(np.array(presults, np.float32), query_list, response_array_refs, good, total, result_timing))
                else:
                    result_queue.put(OItem([], query_list, response_array_refs))

    def run(self):
        os.sched_setaffinity(self.pid, self.affinity)
        torch.set_num_threads(self.core_nums)

        from backend_pytorch_native import get_backend
        backend = get_backend(self.args.backend, self.args.dataset, self.device, self.args.max_ind_range,
                              self.args.data_sub_sample_rate, self.args.use_gpu, self.args.use_ipex, self.args.use_bf16, self.args.use_int8)
        self.model = backend.load(self.args.model_path, self.args.inputs, self.args.outputs)
        print ('Start warmup.')
        self.warmup(self.model)
        print ('Warmup done.')

        self.lock.acquire()
        self.init_counter.value += 1
        self.lock.release()

        from criteo import get_dataset
        ds = get_dataset(self.args)
        sample_list = self.ds_queue.get()
        ds.mlperf_bin_load_query_samples(sample_list)
        self.items_in_memory = ds.items_in_memory
        print(str(self.pid), " : Complete load query samples !!")
        self.lock.acquire()
        self.init_counter.value += 1
        self.lock.release()

        if self.num_ins > 1 :
            for i in range(self.num_ins):
                if self.rqnum == 1:
                    worker = mp.Process(target=self.handle_tasks, args=(i, self.task_queue, self.result_queue[0], self.args, self.pid))
                else:
                    worker = mp.Process(target=self.handle_tasks, args=(i, self.task_queue, self.result_queue[i], self.args, self.pid))
                self.workers.append(worker)
            for w in self.workers:
                w.start()
            for w in self.workers:
                w.join()
        else:
            self.handle_tasks(0, self.task_queue, self.result_queue[0], self.args, self.pid)

        ds.unload_query_samples()
