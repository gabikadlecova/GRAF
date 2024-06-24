from graf_nas.sample.cluster import clustered_data_sample
from graf_nas.sample.random import random_data_sample


sampling_strategies = {
    'random': random_data_sample,
    'cluster': clustered_data_sample
}
