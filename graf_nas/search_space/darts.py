from naslib.search_spaces.nasbench301.conversions import convert_compact_to_genotype


def get_ops_nb301():
    return ["max_pool_3x3", "avg_pool_3x3", "skip_connect", "sep_conv_3x3", "sep_conv_5x5", "dil_conv_3x3",
            "dil_conv_5x5"]


def darts_cell_to_graph(genotype):
    """Adapted from darts plotting script."""
    op_map = ['out', *get_ops_nb301()]
    op_map = {o: i for i, o in enumerate(op_map)}
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
