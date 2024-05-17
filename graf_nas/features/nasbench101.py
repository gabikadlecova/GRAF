import networkx as nx
import numpy as np

from zc_combine.features.base import count_ops_opnodes, min_path_len_opnodes, max_num_on_path_opnodes, get_start_end, \
    get_in_out_edges_opnodes
from zc_combine.features.conversions import to_graph


def node_degree(net, allowed, start=0, end=1):
    _, ops, graph = net
    in_edges, out_edges = get_in_out_edges_opnodes(net, allowed)

    start, end = get_start_end(ops, start, end)
    get_func = lambda x, y: y([len(v) for v in x.values()])

    return {'in_degree': len(in_edges[end]), 'out_degree': len(out_edges[start]), 'avg_in': get_func(in_edges, np.mean),
            'avg_out': get_func(out_edges, np.mean)}


def count_edges(net):
    return len(net[2].edges)


def num_of_paths(net, allowed, start=0, end=1):
    _, ops, graph = net

    edges = [e for e in graph.edges if ops[e[0]] in allowed or ops[e[1]] in allowed]
    graph = to_graph(edges)

    start, end = get_start_end(ops, start, end)
    try:
        path = nx.all_simple_paths(graph, start, end)
    except (nx.NodeNotFound, nx.NetworkXNoPath):
        return 0

    return len([p for p in path])


feature_func_dict = {
    'op_count': count_ops_opnodes,
    'min_path_len': min_path_len_opnodes,
    'max_op_on_path': max_num_on_path_opnodes,
    'node_degree': node_degree
}
