import numpy as np
import torch

from naslib.search_spaces.nasbench301.conversions import convert_compact_to_genotype  # type: ignore

from graf_nas.search_space.base import convert_to_naslib, NetBase, NetGraph
from naslib.search_spaces.nasbench301.encodings import encode_adj  # type: ignore
from naslib.search_spaces.nasbench301.graph import NasBench301SearchSpace, NUM_VERTICES, NUM_OPS  # type: ignore
from typing import Tuple, Dict, List, Iterator


class DARTS(NetBase):
    """
    Respresents a network from the DARTS (and also NAS-Bench-301) search space.
    """

    random_iterator = True  # Returns random architectures from the search space

    def __init__(self, net: str, cache_graph: bool = True):
        """
        Initializes a DARTS network.
        :param net: network string hash (hashable NASLib compact representation)
        """
        super().__init__(net)
        self.cache_graph = cache_graph

    def to_graph(self) -> Tuple[NetGraph, NetGraph]:
        """
        Converts the network to its graph representation - both the normal and reduce cells.
        :return: two network graphs (normal, reduce) - represented by op_names, edges
        """
        return darts_to_graph(self.net, cache_graph=self.cache_graph)

    def to_onehot(self) -> np.ndarray:
        """
        Converts the network to a one-hot encoding.
        :return: one-hot encoding of the network
        """
        return encode_adj(self.net)

    def get_model(self) -> torch.nn.Module:
        """
        Converts the network to a naslib model - a torch module.
        :return: torch model
        """
        return convert_to_naslib(self.net, NasBench301SearchSpace)

    @staticmethod
    def get_op_map() -> Dict[str, int]:
        """
        Returns a mapping of operation names to operation indices. Since each node
        also has a connection to the output node, the 'out' operation is included.

        :return: operation map
        """
        return get_op_map_darts()

    @staticmethod
    def get_arch_iterator(dataset_api=None) -> Iterator['DARTS']:
        """
        Returns an iterator over all architectures in the DARTS search space.
        :param dataset_api: if provided, query the NB301 surrogate api (not supported at the moment)
        :return: iterator over architectures
        """
        if dataset_api is not None:
            raise NotImplementedError("Querying the NAS-Bench-301 surrogate model is not yet supported.")

        while True:
            # from NASLib NB301, without setting the spec (time-consuming step)
            compact_normal, compact_reduce = [], []
            for i in range(NUM_VERTICES):
                ops = np.random.choice(range(NUM_OPS), NUM_VERTICES)

                nodes_in_normal = np.random.choice(range(i + 2), 2, replace=False)
                nodes_in_reduce = np.random.choice(range(i + 2), 2, replace=False)

                compact_normal.extend(
                    [(nodes_in_normal[0], ops[0]), (nodes_in_normal[1], ops[1])]
                )
                compact_reduce.extend(
                    [(nodes_in_reduce[0], ops[2]), (nodes_in_reduce[1], ops[3])]
                )

            # convert the lists to tuples
            compact = tuple(compact_normal), tuple(compact_reduce)

            yield DARTS(str(compact))


def get_op_map_darts() -> Dict[str, int]:
    """
    Returns the operation names and their indices for the DARTS search space.
    """
    op_map = ['out', *get_ops_nb301()]
    return {o: i for i, o in enumerate(op_map)}


def get_ops_nb301() -> List[str]:
    """
    Returns the operation names in the DARTS search space.
    """
    return ["max_pool_3x3", "avg_pool_3x3", "skip_connect", "sep_conv_3x3", "sep_conv_5x5", "dil_conv_3x3",
            "dil_conv_5x5"]


def darts_cell_to_graph(genotype, cache_graph=True):
    """
    Converts a DARTS genotype to a graph representation.
    Adapted from the darts plotting script.
    """
    op_map = get_op_map_darts()
    ops = {i: o for o, i in op_map.items()}
    edges = {}

    assert len(genotype) % 2 == 0
    steps = len(genotype) // 2

    for i in range(steps):
        for k in [2 * i, 2 * i + 1]:
            op, j = genotype[k]
            if j == 0:
                u = "c_{k-2}"
            elif j == 1:
                u = "c_{k-1}"
            else:
                u = str(j - 2)
            v = str(i)
            edges[u, v] = op_map[op]

    for i in range(steps):
        edges[str(i), "c_{k}"] = op_map['out']

    return NetGraph(ops, edges, cache_graph=cache_graph)


def darts_to_graph(net: str, cache_graph=True) -> Tuple[NetGraph, NetGraph]:
    """
    Converts a DARTS network hash to its graph representation
    :param net: DARTS network hash
    :param cache_graph: whether the graph should cache its networkx graph representation
    :return: two network graphs (normal, reduce) - represented by op_names, edges
    """
    if isinstance(net, str):
        genotype = convert_compact_to_genotype(eval(net))
    return darts_cell_to_graph(genotype.normal, cache_graph=cache_graph), darts_cell_to_graph(genotype.reduce, cache_graph=cache_graph)
