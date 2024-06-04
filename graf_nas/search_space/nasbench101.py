import copy
import math
import naslib
import networkx as nx
import numpy as np

from graf_nas.search_space.conversions import convert_to_naslib, NetBase
from naslib.search_spaces.nasbench101.conversions import convert_tuple_to_spec
from naslib.search_spaces.nasbench101.graph import NasBench101SearchSpace


class NB101(NetBase):
    naslib_object = None
    random_iterator = False

    def __init__(self, net):
        super().__init__(net)

    def to_graph(self):
        return nb101_to_graph(self.net)

    def to_onehot(self):
        return nb101_to_onehot(eval(self.net))

    def to_naslib(self):
        return convert_to_naslib(self.net, NasBench101SearchSpace)

    @staticmethod
    def get_op_map():
        return get_op_map_nb101()

    @staticmethod
    def get_arch_iterator(dataset_api):
        if NB101.naslib_object is None:
            NB101.naslib_object = NasBench101SearchSpace()

        for n in NB101.naslib_object.get_arch_iterator(dataset_api):
            yield NB101(str(n))


def get_ops_nb101():
    return ['input', 'output', 'maxpool3x3', 'conv1x1-bn-relu', 'conv3x3-bn-relu']


def get_op_map_nb101():
    op_map = [*get_ops_nb101()]
    return {o: i for i, o in enumerate(op_map)}


def parse_ops_nb101(net, return_edges=True):
    ops = net
    if isinstance(ops, str):
        ops = net.strip('()').split(', ')

    def parse_cell(c):
        n_nodes = int(math.sqrt(len(c)))
        index = n_nodes * n_nodes
        op, edges = c[index:], c[:index]
        assert len(op) == n_nodes
        op = [int(o) for o in op]
        return (op, [int(e) for e in edges]) if return_edges else op

    return parse_cell(ops)


def nb101_to_graph(net):
    op_map = get_op_map_nb101()
    op_map = {i: o for o, i in op_map.items()}

    ops, edges = parse_ops_nb101(net)
    edges = np.array(edges)
    edges = edges.reshape(int(np.sqrt(edges.shape[0])), -1)
    assert edges.shape[0] == edges.shape[1]

    return op_map, ops, nx.from_numpy_array(edges, create_using=nx.DiGraph)


def pad_nb101_net(net):
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


def nb101_to_onehot(net):
    net = convert_tuple_to_spec(net)
    matrix_dim = len(net['matrix'])
    net = pad_nb101_net(net)

    enc = naslib.search_spaces.nasbench101.encodings.encode_adj(net)
    if matrix_dim < 7:
        for i in range(0, 7 - matrix_dim):
            for oid in range(3):
                idx = 3 * i + oid
                enc[-1 - idx] = 0
    return enc
