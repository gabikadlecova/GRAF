import numpy as np
import torch
from naslib.search_spaces.nasbench201.encodings import encode_adjacency_one_hot_op_indices  # type: ignore

from graf_nas.search_space.conversions import convert_to_naslib, NetBase, NetGraph
from naslib.search_spaces.nasbench201.graph import NasBench201SearchSpace  # type: ignore
from typing import List, Tuple, Dict


class NB201(NetBase):
    """
    Respresents a network from the NAS-Bench-201 search space.
    """
    naslib_object = None  # NasBench201SearchSpace object (for architecture iterator)
    random_iterator = False  # Returns all architectures from the benchmark systematically

    def __init__(self, net: str, cache_graph: bool = True):
        """
        Initializes a NAS-Bench-201 network.
        :param net: network string hash (tuple of operation ids on each edge)
        """
        super().__init__(net)
        self.cache_graph = cache_graph

    def to_graph(self) -> NetGraph:
        """
        Converts the network to its graph representation.
        :return: network graph - represented by op_names, edges
        """
        return nb201_to_graph(self.net, cache_graph=self.cache_graph)

    def to_onehot(self) -> np.ndarray:
        """
        Converts the network to a one-hot encoding.
        :return: one-hot encoding of the network
        """
        return encode_adjacency_one_hot_op_indices(eval(self.net))

    def get_model(self) -> torch.nn.Module:
        """
        Converts the network to a naslib model - a torch module.
        :return: torch model
        """
        return convert_to_naslib(self.net, NasBench201SearchSpace)

    @staticmethod
    def get_op_map():
        """
        Returns a mapping of operation names to operation indices.
        :return: operation map
        """
        return {o: i for i, o in enumerate(get_ops_edges_nb201()[0])}

    @staticmethod
    def get_arch_iterator(dataset_api):
        """
        Returns an iterator over all architectures in the NAS-Bench-201 search space.
        For the dataset api, see naslib.utils.get_dataset_api.
        :param dataset_api: NAS-Bench-201 dataset api
        :return: iterator over architectures
        """
        if NB201.naslib_object is None:
            NB201.naslib_object = NasBench201SearchSpace()

        for n in NB201.naslib_object.get_arch_iterator(dataset_api):
            yield NB201(str(n))


def get_ops_edges_nb201():
    """
    Returns the operation names and edge names for the NAS-Bench-201 search space.
    The edges have a specific order, and the operation positions determine their ids.

    :return: operation names, edge names
    """
    edge_map = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
    edge_map = {val: i for i, val in enumerate(edge_map)}
    ops = ['skip_connect', 'none', 'nor_conv_3x3', 'nor_conv_1x1', 'avg_pool_3x3']
    return ops, edge_map

def nb201_like_to_graph(net: Tuple[int, ...], ops: List[str], edge_map: Dict[Tuple[int, int], int], cache_graph: bool = False) -> NetGraph:
    """
    Converts a NAS-Bench-201 tuple hash to a graph representation.
    :param net: network string hash (tuple of operation ids on each edge)
    :param ops: operation names
    :param edge_map: edge names
    :param cache_graph: whether the graph should cache its networkx graph representation
    :return: network graph - op_names, edges
    """
    op_map = {i: op for i, op in enumerate(ops)}
    edges = {k: net[i] for k, i in edge_map.items()}
    return NetGraph(op_map, edges, cache_graph=cache_graph)


def parse_ops_nb201(net: str) -> Tuple[int, ...]:
    """
    Parses the network string hash to a list of operation indices.
    :param net: network string hash
    :return: list of operation indices
    """
    ops = net.strip('()').split(', ') if isinstance(net, str) else net
    return tuple([int(op) for op in ops])


def nb201_to_graph(net: str, cache_graph=True) -> NetGraph:
    """
    Converts a NAS-Bench-201 NASLib hash to a graph representation.
    :param net: network string hash
    :param cache_graph: whether the graph should cache its networkx graph representation
    :return: network graph - op_names, edges
    """
    op_map = parse_ops_nb201(net)
    ops, edges = get_ops_edges_nb201()
    return nb201_like_to_graph(op_map, ops, edges, cache_graph=cache_graph)
