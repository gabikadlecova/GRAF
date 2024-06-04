from naslib.search_spaces.transbench101.encodings import encode_adjacency_one_hot_transbench_micro_op_indices
from naslib.search_spaces.transbench101.graph import TransBench101SearchSpaceMicro

from graf_nas.search_space.conversions import convert_to_naslib, NetBase
from graf_nas.search_space.nasbench201 import parse_ops_nb201, nb201_like_to_graph


class TNB101_micro(NetBase):
    naslib_object = None
    random_iterator = False

    def __init__(self, net):
        super().__init__(net)

    def to_graph(self):
        return tnb101_to_graph(self.net)

    def to_onehot(self):
        return encode_adjacency_one_hot_transbench_micro_op_indices(eval(self.net))

    def to_naslib(self):
        return convert_to_naslib(self.net, TransBench101SearchSpaceMicro)

    @staticmethod
    def get_op_map():
        return {o: i for i, o in enumerate(get_ops_edges_tnb101()[0])}

    @staticmethod
    def get_arch_iterator(dataset_api):
        if TNB101_micro.naslib_object is None:
            TNB101_micro.naslib_object = TransBench101SearchSpaceMicro()

        for n in TNB101_micro.naslib_object.get_arch_iterator(dataset_api):
            yield TNB101_micro(str(n))


def get_ops_edges_tnb101():
    edge_map = ((1, 2), (1, 3), (2, 3), (1, 4), (2, 4), (3, 4))
    edge_map = {val: i for i, val in enumerate(edge_map)}

    ops = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
    return ops, edge_map


def tnb101_to_graph(net):
    net = parse_ops_nb201(net)
    ops, edges = get_ops_edges_tnb101()
    return nb201_like_to_graph(net, ops, edges)
