import numpy as np
import torch
from graf_nas.search_space.base import convert_to_naslib, NetBase

from naslib.search_spaces.transbench101.encodings import encode_adjacency_one_hot_transbench_macro_op_indices  # type: ignore
from naslib.search_spaces.transbench101.graph import TransBench101SearchSpaceMacro  # type: ignore
from typing import Dict, List, Iterator


class TNB101MacroGraph:
    """
    Represents a sequential architecture where ops are a convolutional layers.
    Each op can have a stride of 1 or 2 and may have a channel increase.
    """
    def __init__(self, ops: List[Dict[str, bool]]):
        self.ops = ops

    def __iter__(self):
        return iter(self.ops)


class TNB101_macro(NetBase):
    """
    Respresents a network from the TransBench-101 search space.
    """
    naslib_object = None  # TransBench101SearchSpaceMacro object (for architecture iterator)
    random_iterator = False  # Returns all architectures from the benchmark systematically   

    def __init__(self, net: str):
        """
        Initializes a TransBench-101 network.
        :param net: network string hash (tuple of sequential operation ids)
        """
        super().__init__(net)

    def to_graph(self) -> TNB101MacroGraph:
        """
        Converts the network to its graph representation.
        :return: network graph - list of dicts with channel increase and stride indicators
        """
        return tnb101_macro_encode(self.net)

    def to_onehot(self) -> np.ndarray:
        """
        Converts the network to a one-hot encoding.
        :return: one-hot encoding of the network
        """
        return encode_adjacency_one_hot_transbench_macro_op_indices(eval(self.net))

    def get_model(self) -> torch.nn.Module:
        """
        Converts the network to a naslib model - a torch module.
        """
        return convert_to_naslib(self.net, TransBench101SearchSpaceMacro)

    @staticmethod
    def get_arch_iterator(dataset_api) -> Iterator['TNB101_macro']:
        """
        Returns an iterator over all architectures in the TransBench-101 search space.
        For the dataset api, see naslib.utils.get_dataset_api.
        :param dataset_api: TransBench-101 dataset api
        :return: iterator over architectures
        """
        if TNB101_macro.naslib_object is None:
            TNB101_macro.naslib_object = TransBench101SearchSpaceMacro()

        for n in TNB101_macro.naslib_object.get_arch_iterator(dataset_api):
            yield TNB101_macro(str(n))


def tnb101_macro_encode(net):
    """
    Converts a network hash to a TNB101MacroGraph object.
    :param net: network hash (tuple of sequential operation ids)
    :return: TNB101MacroGraph object
    """
    if isinstance(net, str):
        net = net.strip('()').split(',')

    res = []

    for idx in net:
        encoding = {}
        idx = int(idx)
        encoding['channel'] = idx % 2 == 0
        encoding['stride'] = idx > 2
        res.append(encoding)

    return TNB101MacroGraph(res)
