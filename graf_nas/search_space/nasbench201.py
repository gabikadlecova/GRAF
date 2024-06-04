from naslib.search_spaces.nasbench201.encodings import encode_adjacency_one_hot_op_indices

from graf_nas.search_space.conversions import convert_to_naslib, NetBase
from naslib.search_spaces.nasbench201.graph import NasBench201SearchSpace


class NB201(NetBase):
    naslib_object = None
    random_iterator = False

    def __init__(self, net):
        super().__init__(net)

    def to_graph(self):
        return nb201_to_graph(self.net)

    def to_onehot(self):
        return encode_adjacency_one_hot_op_indices(eval(self.net))

    def to_naslib(self):
        return convert_to_naslib(self.net, NasBench201SearchSpace)

    @staticmethod
    def get_op_map():
        return {o: i for i, o in enumerate(get_ops_edges_nb201()[0])}

    @staticmethod
    def get_arch_iterator(dataset_api):
        if NB201.naslib_object is None:
            NB201.naslib_object = NasBench201SearchSpace()

        for n in NB201.naslib_object.get_arch_iterator(dataset_api):
            yield NB201(str(n))


def get_ops_edges_nb201():
    edge_map = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
    edge_map = {val: i for i, val in enumerate(edge_map)}
    ops = ['skip_connect', 'none', 'nor_conv_3x3', 'nor_conv_1x1', 'avg_pool_3x3']
    return ops, edge_map


def nb201_like_to_graph(net, ops, edge_map):
    ops = {i: op for i, op in enumerate(ops)}
    edges = {k: net[i] for k, i in edge_map.items()}
    return ops, edges


def parse_ops_nb201(net):
    ops = net.strip('()').split(', ') if isinstance(net, str) else net
    return [int(op) for op in ops]


def nb201_to_graph(net):
    net = parse_ops_nb201(net)
    ops, edges = get_ops_edges_nb201()
    return nb201_like_to_graph(net, ops, edges)
