def get_ops_edges_nb201():
    edge_map = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
    edge_map = {val: i for i, val in enumerate(edge_map)}
    ops = ['skip_connect', 'none', 'nor_conv_3x3', 'nor_conv_1x1', 'avg_pool_3x3']
    return ops, edge_map


def get_ops_edges_tnb101():
    edge_map = ((1, 2), (1, 3), (2, 3), (1, 4), (2, 4), (3, 4))
    edge_map = {val: i for i, val in enumerate(edge_map)}

    ops = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
    return ops, edge_map


def _nb201_like_to_graph(net, ops, edge_map):
    ops = {i: op for i, op in enumerate(ops)}
    edges = {k: net[i] for k, i in edge_map.items()}
    return ops, edges


def parse_ops_nb201(net):
    ops = net.strip('()').split(', ') if isinstance(net, str) else net
    return [int(op) for op in ops]


def nb201_to_graph(net):
    net = parse_ops_nb201(net)
    ops, edges = get_ops_edges_nb201()
    return _nb201_like_to_graph(net, ops, edges)


def tnb101_to_graph(net):
    net = parse_ops_nb201(net)
    ops, edges = get_ops_edges_tnb101()
    return _nb201_like_to_graph(net, ops, edges)
