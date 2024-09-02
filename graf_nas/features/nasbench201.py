import numpy as np

from graf_nas.features.base import count_ops, min_path_len, max_num_on_path, get_in_out_edges, NetGraph
from typing import List, Dict


def node_degree(net: NetGraph, allowed: List[int], start: int = 1, end: int = 4):
    in_edges, out_edges = get_in_out_edges(net, allowed)

    get_avg = lambda x: np.mean([len(v) for v in x.values()]).astype(float)

    return {'in_degree': len(in_edges[start]), 'out_degree': len(out_edges[end]), 'avg_in': get_avg(in_edges),
            'avg_out': get_avg(out_edges)}


feature_func_dict = {
    'op_count': count_ops,
    'min_path_len': min_path_len,
    'max_op_on_path': max_num_on_path,
    'node_degree': node_degree
}
