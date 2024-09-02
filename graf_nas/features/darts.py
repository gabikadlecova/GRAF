import numpy as np

from graf_nas.features.base import count_ops, get_in_out_edges, max_num_on_path, min_path_len
from graf_nas.search_space.base import NetGraph
from collections.abc import Callable
from typing import List, Dict, Tuple, Any


def _get_for_both_cells(cells: Tuple[NetGraph, NetGraph], func: Callable[[NetGraph], Any]) -> Dict[str, Any]:
    normal = func(cells[0])
    reduce = func(cells[1])

    if isinstance(normal, dict):
        return {**{f"normal_{k}": v for k, v in normal.items()}, **{f"reduce_{k}": v for k, v in reduce.items()}}

    return {'normal': normal, 'reduce': reduce}


def get_op_counts(cells: Tuple[NetGraph, NetGraph]) -> Dict[str, int]:
    return _get_for_both_cells(cells, count_ops)


def get_special_nodes() -> List[str]:
    return ["c_{k-2}", "c_{k-1}", "c_{k}"]


def _get_cell_degrees(net: NetGraph, allowed: List[int]) -> Dict[str, float]:
    get_avg = lambda x: np.mean([len(v) for v in x.values()]).astype(float)

    input1, input2, _ = get_special_nodes()

    in_edges, out_edges = get_in_out_edges(net, allowed)
    return {f'{input1}_degree': len(in_edges[input1]), f'{input2}_degree': len(in_edges[input2]),
            'avg_in': get_avg(in_edges), 'avg_out': get_avg(out_edges)}


def get_node_degrees(cells: Tuple[NetGraph, NetGraph], allowed: List[int]) -> Dict[str, int]:
    return _get_for_both_cells(cells, lambda c: _get_cell_degrees(c, allowed))


def _get_both_max_paths(net: NetGraph, allowed: List[Any]) -> Dict[str, int]:
    # last connection (edge) from any node to the output is the 'out' operation
    if 'out' not in allowed:
        if isinstance(allowed, set):
            allowed.add('out')
        else:
            allowed = [*allowed, 'out']

    input1, input2, output = get_special_nodes()

    res1 = max_num_on_path(net, allowed, start=input1, end=output)
    res2 = max_num_on_path(net, allowed, start=input2, end=output)
        
    # if a path is found, omit connection from the last node to the output
    def adjust_by_one(r):
        return r - 1 if len(r) > 0 else 0

    return {f"{input1}": adjust_by_one(res1), f"{input2}": adjust_by_one(res2)}


def get_max_path(cells: Tuple[NetGraph, NetGraph], allowed: List[int]) -> Dict[str, int]:
    return _get_for_both_cells(cells, lambda c: _get_both_max_paths(c, allowed))


def _get_all_min_paths(net: NetGraph, allowed: List[int]) -> Dict[str, int]:
    special_nodes = get_special_nodes()
    all_nodes = set()
    for e in net.edges:
        all_nodes.add(e[0])
        all_nodes.add(e[1])

    def get_lengths(start_node: str | int) -> Dict[str, int]:
        res = {}
        for node in all_nodes:
            if node in special_nodes:
                continue
            res[str((start_node, node))] = min_path_len(net, allowed, start=start_node, end=node, max_val=len(all_nodes))
        return res

    return {**get_lengths(special_nodes[0]), **get_lengths(special_nodes[1])}


def get_min_node_path(cells: Tuple[NetGraph, NetGraph], allowed: List[int]) -> Dict[str, int]:
    """Compared to min path len in nb201, every node is connected to the output and so we look at intermediate nodes
       to assess depth."""
    return _get_for_both_cells(cells, lambda c: _get_all_min_paths(c, allowed))


feature_func_dict = {
    'op_count': get_op_counts,
    'node_degree': get_node_degrees,
    'max_op_on_path': get_max_path
}
