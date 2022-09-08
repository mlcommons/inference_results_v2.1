import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("SQuAD")

class Dataset:
    def __init__(self, dataset_param, mlperf_param):
        self.output_list = []
        self.input_list_inmemory = {}
        self.output_list_inmemory = {}
        self.count = mlperf_param["total_sample_count"]
        self.batch_size = dataset_param["batch_size"]
        self.max_seq_length = dataset_param["max_seq_length"]
        self.doc_stride = dataset_param["doc_stride"]
        self.vocab_path = dataset_param["vocab_path"]
        self.cache_dir = dataset_param["cache_dir"]
        logging.info(f"example count is {self.count}")

    def load_query_samples(self, sample_index_list):
        """
        Called by loadgen to load samples before sending queries to sut.
        Ideally complementary to load_dataset. If using this to load samples by loadgen, the samples are not necessarily available across processes - Needs to figure out if possible to work this out
        """
        pass

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
        pass

    def get_samples(self, sample_index_list=[]):
        """
        Fetches and returns pre-processed data at requested 'sample_index_list'
        """
        pass

    def get_warmup_samples(self):
        """
        Fetches and returns pre-processed data for warmup
        """
        pass

    def pre_process(self, file=""):
        """
        Pre-processes a given input/image
        """
        pass

    def post_process(self, query_ids, sample_index_list, results):
        """
        Post-processor that accepts loadgens query ids and corresponding inference output.
        post_process should return and OutputItem object which has two attributes:
        OutputItem.query_id_list
        OutputItem.results
        """
        return results
