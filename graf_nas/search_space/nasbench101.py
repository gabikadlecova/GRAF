import copy
import math
import naslib  # type: ignore
import networkx as nx
import numpy as np
import torch

from graf_nas.search_space.base import convert_to_naslib, NetBase, OpNodesNetGraph
from naslib.search_spaces.nasbench101.conversions import convert_tuple_to_spec, convert_spec_to_tuple  # type: ignore
from naslib.search_spaces.nasbench101.graph import NasBench101SearchSpace  # type: ignore
from typing import Dict, List, Tuple, Iterator


class NB101(NetBase):
    """
    Respresents a network from the NAS-Bench-101 search space.
    """
    naslib_object = None  # NasBench101SearchSpace object (for architecture iterator)
    random_iterator = False  # Returns all architectures from the benchmark systematically

    def __init__(self, net: str):
        """
        Initializes a NAS-Bench-101 network.
        :param net: network string hash (flattened adjacency matrix and list of operations)
        """
        super().__init__(net)

    def to_graph(self) -> OpNodesNetGraph:
        """
        Converts the network to its graph representation.
        :return: network graph - op_names, op_ids, networkx graph
        """
        return nb101_to_graph(self.net)

    def to_onehot(self) -> np.ndarray:
        """
        Converts the network to a one-hot encoding.
        :return: one-hot encoding of the network
        """
        return nb101_to_onehot(eval(self.net))

    def get_model(self) -> torch.nn.Module:
        """
        Converts the network to a naslib model - a torch module.
        :return: torch model
        """
        return convert_to_naslib(self.net, NasBench101SearchSpace)

    @staticmethod
    def get_op_map() -> Dict[str, int]:
        """
        Returns a mapping of operation names to operation indices.
        :return: operation map
        """
        return get_op_map_nb101()
    
    @staticmethod
    def str_hash_to_tuple_hash(net_hash: str, dataset_api) -> str:
        """
        Converts a NAS-Bench-101 network hash to a NASLib tuple hash. The dataset api
        can be either dataset_api["nb101_data"] from naslib.utils.get_dataset_api, or
        the original NAS-Bench-101 dataset api.

        :param net_hash: network hash from NAS-Bench-101
        :param dataset_api: dataset api
        :return: network tuple hash from NASLib
        """
        spec, _ = dataset_api.get_metrics_from_hash(net_hash)
        spec = {'matrix': spec['module_adjacency'], 'ops': spec['module_operations']}
        return str(convert_spec_to_tuple(spec))

    @staticmethod
    def get_arch_iterator(dataset_api) -> Iterator['NB101']:
        """
        Returns an iterator over NAS-Bench-101 architectures.
        :param dataset_api: NASLib dataset api (from naslib.utils.get_dataset_api)
        """
        if NB101.naslib_object is None:
            NB101.naslib_object = NasBench101SearchSpace()

        for n in NB101.naslib_object.get_arch_iterator(dataset_api):
            # the dataset API returns NB101 hashes, we use the NASLib tuple hash
            n = NB101.str_hash_to_tuple_hash(n, dataset_api["nb101_data"])
            yield NB101(n)


def get_ops_nb101():
    """
    Returns the operation names in the NAS-Bench-101 search space.
    """
    return ['input', 'output', 'maxpool3x3', 'conv1x1-bn-relu', 'conv3x3-bn-relu']


def get_op_map_nb101() -> Dict[str, int]:
    """
    Returns a mapping of operation names to operation indices.
    """
    op_map = [*get_ops_nb101()]
    return {o: i for i, o in enumerate(op_map)}


def _parse_ops_nb101(net: str) -> Tuple[List[int], List[int]]:
    """
    Parses the operations and edges from a NAS-Bench-101 network hash.
    The hash is a flattened adjacency matrix concatenated with a list of operations.
    :param net: network hash
    :return: operations, flattened edges - both as lists of ids
    """
    ops = net.strip('()').split(', ')

    # len(c) = n * n + n
    # n < sqrt(n * n + n) < n + 1 ... floor(sqrt(len(c))) == n
    n_nodes = math.floor(math.sqrt(len(net)))
    index = n_nodes * n_nodes
    op, edges = ops[index:], ops[:index]
    assert len(op) == n_nodes
    op_list = [int(o) for o in op]
    return op_list, [int(e) for e in edges]


def nb101_to_graph(net: str) -> OpNodesNetGraph:
    """
    Converts a NAS-Bench-101 network hash to a networkx graph.
    :param net: network hash
    """
    op_map = get_op_map_nb101()
    rev_op_map = {i: o for o, i in op_map.items()}

    op_ids, edges = _parse_ops_nb101(net)
    adj = np.array(edges)
    adj = adj.reshape(int(np.sqrt(adj.shape[0])), -1)
    assert adj.shape[0] == adj.shape[1]

    return OpNodesNetGraph(rev_op_map, op_ids, nx.from_numpy_array(adj, create_using=nx.DiGraph))


def pad_nb101_spec(net: dict) -> dict:
    """
    Pads a NAS-Bench-101 spec to have 7 nodes.
    :param net: network spec
    :return: padded network spec
    """
    matrix_dim = len(net['matrix'])
    if matrix_dim < 7:
        net = copy.deepcopy(net)
        padval = 7 - matrix_dim
        net['matrix'] = np.pad(net['matrix'], [(0, padval), (0, padval)])
        net['matrix'][:, -1] = net['matrix'][:, -(padval + 1)]
        net['matrix'][:, -(padval + 1)] = 0
        for _ in range(padval):
            net['ops'].insert(-1, 'maxpool3x3')

    return net


def nb101_to_onehot(net: Tuple[int]) -> np.ndarray:
    """
    Converts a NAS-Bench-101 network tuple to a one-hot encoding.
    :param net: network tuple (of ints)
    :return: one-hot encoding
    """
    spec = convert_tuple_to_spec(net)
    matrix_dim = len(spec['matrix'])
    spec = pad_nb101_spec(spec)

    enc = naslib.search_spaces.nasbench101.encodings.encode_adj(net)
    if matrix_dim < 7:
        for i in range(0, 7 - matrix_dim):
            for oid in range(3):
                idx = 3 * i + oid
                enc[-1 - idx] = 0
    return enc
