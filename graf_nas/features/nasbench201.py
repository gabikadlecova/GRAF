import numpy as np

from graf_nas.features.base import count_ops, min_path_len, max_num_on_path, get_in_out_edges


def node_degree(net, allowed, start=1, end=4):
    _, edges = net
    in_edges, out_edges = get_in_out_edges(edges, allowed)

    get_avg = lambda x: np.mean([len(v) for v in x.values()])

    return {'in_degree': len(in_edges[start]), 'out_degree': len(out_edges[end]), 'avg_in': get_avg(in_edges),
            'avg_out': get_avg(out_edges)}


feature_func_dict = {
    'op_count': count_ops,
    'min_path_len': min_path_len,
    'max_op_on_path': max_num_on_path,
    'node_degree': node_degree
}
