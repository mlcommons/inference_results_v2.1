class InputItem:
    def __init__(self, id_list, index_list, data=None, label=None, receipt_time=0):
        self.query_id_list = id_list
        self.sample_index_list = index_list
        self.data = data
        self.label = label
        self.receipt_time = receipt_time


class InputData:
    def __init__(self, data=None, data_shape=None, sequence_len=None):
        self.data = data
        self.input_shape = data_shape
        self.sequence_len = sequence_len


class OutputItem:
    def __init__(self, query_id_list, result, array_type_code='B'):
        self.query_id_list = query_id_list
        self.result = result
        self.array_type_code = array_type_code
        self.receipt_time = None
        self.outqueue_time = None

    def set_receipt_time(self, receipt_time):
        self.receipt_time = receipt_time

    def set_outqueued_time(self, outqueue_time):
        self.outqueue_time = outqueue_time


class BaseInQueue:
    def __init__(self, mpQueue=None, batch_size=1, **kwargs):
        self.in_queue = mpQueue
        self.batch_size = batch_size

    def put(self, query_samples, receipt_time):
        """
        Receives query sample(s) from loadgen and processes the queries in batches if required/desired.
        Processed/batched queries are wrapped in 'InputItem' object, and then placed on the producer queue
        """
        num_samples = len(query_samples)
        if num_samples == 1:
            item = InputItem([query_samples[0].id], [query_samples[0].index], receipt_time=receipt_time)
            self.in_queue.put(item)

        else:
            #idx = [q.index for q in query_samples]
            #query_id = [q.id for q in query_samples]

            num_batches = num_samples // self.batch_size
            remainder = num_samples % self.batch_size
            batch = 0
            bidx = 0
            bs = self.batch_size
            while batch < num_batches:
                j = 0
                ids = []
                indexes = []
                while j < bs:
                    ids.append(query_samples[bidx].id)
                    indexes.append(query_samples[bidx].index)
                    bidx += 1
                    j += 1

                item = InputItem(ids, indexes, receipt_time=receipt_time)
                self.in_queue.put( item )
                batch += 1

            ids = []
            indexes = []
            while bidx < num_samples:
                ids.append(query_samples[bidx].id)
                indexes.append(query_samples[bidx].index)
                bidx += 1

                item = InputItem(ids, indexes, receipt_time=receipt_time)
                self.in_queue.put( item )


class BaseOutQueue:
    pass
