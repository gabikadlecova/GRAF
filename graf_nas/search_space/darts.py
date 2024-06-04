import numpy as np

from naslib.search_spaces.nasbench301.conversions import convert_compact_to_genotype

from graf_nas.search_space.conversions import convert_to_naslib, NetBase
from naslib.search_spaces.nasbench301.encodings import encode_adj
from naslib.search_spaces.nasbench301.graph import NasBench301SearchSpace, NUM_VERTICES, NUM_OPS


class DARTS(NetBase):
    random_iterator = True

    def __init__(self, net):
        super().__init__(net)

    def to_graph(self):
        return darts_to_graph(self.net)

    def to_onehot(self):
        return encode_adj(self.net)

    def to_naslib(self):
        return convert_to_naslib(self.net, NasBench301SearchSpace)

    @staticmethod
    def get_op_map():
        return get_op_map_darts()

    @staticmethod
    def get_arch_iterator(dataset_api=None):
        while True:
            # from NASLib NB301, without setting the spec (time-consuming step)
            compact = [[], []]
            for i in range(NUM_VERTICES):
                ops = np.random.choice(range(NUM_OPS), NUM_VERTICES)

                nodes_in_normal = np.random.choice(range(i + 2), 2, replace=False)
                nodes_in_reduce = np.random.choice(range(i + 2), 2, replace=False)

                compact[0].extend(
                    [(nodes_in_normal[0], ops[0]), (nodes_in_normal[1], ops[1])]
                )
                compact[1].extend(
                    [(nodes_in_reduce[0], ops[2]), (nodes_in_reduce[1], ops[3])]
                )

            # convert the lists to tuples
            compact[0] = tuple(compact[0])
            compact[1] = tuple(compact[1])
            compact = tuple(compact)

            yield DARTS(str(compact))


def get_op_map_darts():
    op_map = ['out', *get_ops_nb301()]
    return {o: i for i, o in enumerate(op_map)}


def get_ops_nb301():
    return ["max_pool_3x3", "avg_pool_3x3", "skip_connect", "sep_conv_3x3", "sep_conv_5x5", "dil_conv_3x3",
            "dil_conv_5x5"]


def darts_cell_to_graph(genotype):
    """Adapted from darts plotting script."""
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

    return ops, edges


def darts_to_graph(net):
    if isinstance(net, str):
        net = convert_compact_to_genotype(eval(net))
    return darts_cell_to_graph(net.normal), darts_cell_to_graph(net.reduce)
