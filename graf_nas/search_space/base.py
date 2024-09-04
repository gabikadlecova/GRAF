import networkx as nx  # type: ignore
import numpy as np
from abc import ABC, abstractmethod
import torch
from typing import Dict, List, Tuple, Optional, Any


class NetBase(ABC):
    def __init__(self, net: str, cache_model: bool = False):
        self.net = net
        self.cache_model = cache_model
        self.model: torch.nn.Module | None = None

    def get_hash(self) -> str:
        return self.net

    @abstractmethod
    def to_graph(self):
        raise NotImplementedError()

    @abstractmethod
    def to_onehot(self) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def create_model(self) -> torch.nn.Module:
        pass

    def get_model(self) -> torch.nn.Module:
        # retrieve cached model if available
        if self.cache_model and self.model is not None:
            return self.model
        
        # create torch model, optionally cache
        model = self.create_model()
        if self.cache_model:
            self.model = model
        return model


class NetGraph:
    """
    Represents a network architecture where operations are stored on edges.

    The network is represented with a ops and edges.
    - ops - the operation map is a dict of operation ids mapped to operation names.
    - edges - a dict of edge names (pair of connections (node, node)) mapped to operation ids.
    """
    def __init__(self, ops: Dict[int, str], edges: Dict[Tuple[Any, Any], int], cache_graph: bool = True):
        """
        :param ops: dict of operation ids mapped to operation names
        :param edges: dict of edge names mapped to operation ids
        :param cache_graph: whether to cache the networkx graph representation of the network
        """
        self.ops = ops
        self.edges = edges
        self.cache_graph = cache_graph
        self._graph: Optional[nx.DiGraph] = None

    def to_graph(self) -> nx.DiGraph:
        """
        Converts the network to a networkx graph representation.
        :return: networkx graph
        """
        if self._graph is not None:
            return self._graph

        G: nx.DiGraph = nx.DiGraph()
        for e_from, e_to in self.edges.keys():
            G.add_edge(e_from, e_to)

        if self.cache_graph:
            self._graph = G

        return G


class OpNodesNetGraph:
    """
    Represents a network architecture where operations are stored on nodes.

    The network is represented with ops, op_ids and a networkx graph.
    - ops - the operation map is a dict of operation ids mapped to operation names.
    - op_ids - a list of operation indices at nodes.
    - graph - a networkx graph representation of the network.
    """
    def __init__(self, ops: Dict[int, str], op_ids: List[int], graph: nx.DiGraph):
        """
        :param ops: dict of operation ids mapped to operation names
        :param op_ids: list of operation indices at nodes
        :param graph: networkx graph representation of the network
        """
        self.ops = ops
        self.op_ids = op_ids
        self.graph = graph

    def to_graph(self) -> nx.DiGraph:
        return self.graph


def convert_to_naslib(net, naslib_object, **kwargs):
    """
    Converts a network architecture to a naslib object.
    :param net: network architecture - the network hash
    :param naslib_object: naslib object to convert to (a search space)
    :param kwargs: additional arguments to pass to the naslib object
    """
    if isinstance(net, str):
        net = eval(net)

    naslib_obj = naslib_object(**kwargs)
    naslib_obj.set_spec(net)
    naslib_obj.parse()
    return naslib_obj
