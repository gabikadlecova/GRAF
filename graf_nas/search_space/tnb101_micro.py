import numpy as np
import torch
import warnings

from naslib.search_spaces.transbench101.encodings import encode_adjacency_one_hot_transbench_micro_op_indices  # type: ignore
from naslib.search_spaces.transbench101.graph import TransBench101SearchSpaceMicro  # type: ignore

from graf_nas.search_space.conversions import convert_to_naslib, NetBase, NetGraph
from graf_nas.search_space.nasbench201 import parse_ops_nb201, nb201_like_to_graph
from typing import Dict, List, Tuple, Iterator


class TNB101_micro(NetBase):
    """
    Respresents a network from the TransBench-101 search space.
    """
    naslib_object = None  # TransBench101SearchSpaceMicro object (for architecture iterator)
    random_iterator = False  # Returns all architectures from the benchmark systematically

    def __init__(self, net: str, cache_graph: bool = True):
        """
        Initializes a TransBench-101 network.
        :param net: network string hash (tuple of operation ids on each edge)
        """
        super().__init__(net)
        self.cache_graph = cache_graph

    def to_graph(self) -> NetGraph:
        """
        Converts the network to its graph representation.
        :return: network graph - represented by op_names, edges
        """
        return tnb101_to_graph(self.net, cache_graph=self.cache_graph)

    def to_onehot(self) -> np.ndarray:
        """
        Converts the network to a one-hot encoding.
        :return: one-hot encoding of the network
        """
        return encode_adjacency_one_hot_transbench_micro_op_indices(eval(self.net))

    def get_model(self) -> torch.nn.Module:
        """
        Converts the network to a naslib model - a torch module.
        """
        return convert_to_naslib(self.net, TransBench101SearchSpaceMicro)

    @staticmethod
    def get_op_map() -> Dict[str, int]:
        """
        Returns a mapping of operation names to operation indices.
        :return: operation map
        """
        return {o: i for i, o in enumerate(get_ops_edges_tnb101()[0])}

    @staticmethod
    def get_arch_iterator(dataset_api) -> Iterator['TNB101_micro']:
        """
        Returns an iterator over all architectures in the TransBench-101 search space.
        For the dataset api, see naslib.utils.get_dataset_api.
        :param dataset_api: TransBench-101 dataset api
        :return: iterator over architectures
        """
        if TNB101_micro.naslib_object is None:
            TNB101_micro.naslib_object = TransBench101SearchSpaceMicro()

        for n in TNB101_micro.naslib_object.get_arch_iterator(dataset_api):
            yield TNB101_micro(str(n))


def get_ops_edges_tnb101():
    """
    Returns the operation names and edge names for the TransBench-101 search space.
    """
    warnings.warn("There is a mismatch in encodings in NASLib. Be cautius when using this search space and wait for future updates.")
    edge_map = ((1, 2), (1, 3), (2, 3), (1, 4), (2, 4), (3, 4))
    edge_map = {val: i for i, val in enumerate(edge_map)}

    ops = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
    return ops, edge_map


def tnb101_to_graph(net, cache_graph=True) -> NetGraph:
    """
    Converts the TransBench-101 network to its graph representation.
    """
    net = parse_ops_nb201(net)
    ops, edges = get_ops_edges_tnb101()
    return nb201_like_to_graph(net, ops, edges, cache_graph=cache_graph)
