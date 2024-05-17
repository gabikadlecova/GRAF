import numpy as np

from zc_combine.features.base import count_ops, get_in_out_edges, max_num_on_path, min_path_len


def _get_for_both_cells(cells, func):
    normal = func(cells[0])
    reduce = func(cells[1])

    if isinstance(normal, dict):
        return {**{f"normal_{k}": v for k, v in normal.items()}, **{f"reduce_{k}": v for k, v in reduce.items()}}

    return {'normal': normal, 'reduce': reduce}


def get_op_counts(cells, to_names=False):
    return _get_for_both_cells(cells, lambda c: count_ops(c, to_names=to_names))


def get_special_nodes():
    return ["c_{k-2}", "c_{k-1}", "c_{k}"]


def _get_cell_degrees(net, allowed):
    get_avg = lambda x: np.mean([len(v) for v in x.values()])

    input1, input2, output = get_special_nodes()

    in_edges, out_edges = get_in_out_edges(net[1], allowed)
    return {f'{input1}_degree': len(in_edges[input1]), f'{input2}_degree': len(in_edges[input2]),
            'avg_in': get_avg(in_edges), 'avg_out': get_avg(out_edges)}


def get_node_degrees(cells, allowed):
    return _get_for_both_cells(cells, lambda c: _get_cell_degrees(c, allowed))


def _get_both_max_paths(net, allowed):
    if 'out' not in allowed:
        if isinstance(allowed, set):
            allowed.add('out')
        else:
            allowed = [*allowed, 'out']

    input1, input2, output = get_special_nodes()

    res1 = max_num_on_path(net, allowed, start=input1, end=output)
    res2 = max_num_on_path(net, allowed, start=input2, end=output)
    return {f"{input1}": res1, f"{input2}": res2}


def get_max_path(cells, allowed):
    return _get_for_both_cells(cells, lambda c: _get_both_max_paths(c, allowed))


def _get_all_min_paths(net, banned):
    special_nodes = get_special_nodes()
    all_nodes = set()
    for e in net[1]:
        all_nodes.add(e[0])
        all_nodes.add(e[1])

    def get_lengths(start_node):
        res = {}
        for node in all_nodes:
            if node in special_nodes:
                continue
            res[str((start_node, node))] = min_path_len(net, banned, start=start_node, end=node, max_val=len(all_nodes))
        return res

    return {**get_lengths(special_nodes[0]), **get_lengths(special_nodes[1])}


def get_min_node_path(cells, banned):
    """Compared to min path len in nb201, every node is connected to the output and so we look at intermediate nodes
       to assess depth."""
    return _get_for_both_cells(cells, lambda c: _get_all_min_paths(c, banned))


feature_func_dict = {
    'op_count': get_op_counts,
    'node_degree': get_node_degrees,
    'max_op_on_path': get_max_path
}
