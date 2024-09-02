import networkx as nx
import numpy as np

from graf_nas.features.base import count_ops_opnodes, min_path_len_opnodes, max_num_on_path_opnodes, get_start_end, \
    get_in_out_edges_opnodes, to_graph, OpNodesNetGraph
from typing import List, Dict


def node_degree(net: OpNodesNetGraph, allowed: List[int], start: int = 0, end: int = 1) -> Dict[str, float]:
    ops = net.op_ids
    in_edges, out_edges = get_in_out_edges_opnodes(net, allowed)

    start, end = get_start_end(ops, start, end)
    get_func = lambda x, y: y([len(v) for v in x.values()])

    return {'in_degree': len(in_edges[end]), 'out_degree': len(out_edges[start]), 'avg_in': get_func(in_edges, np.mean),
            'avg_out': get_func(out_edges, np.mean)}


def count_edges(net: OpNodesNetGraph) -> int:
    return len(net.graph.edges)


def num_of_paths(net: OpNodesNetGraph, allowed: List[int], start: int = 0, end: int = 1) -> int:
    ops, graph = net.op_ids, net.graph

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
