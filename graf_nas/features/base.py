import networkx as nx
from graf_nas.search_space.base import NetGraph, OpNodesNetGraph
from collections.abc import Callable
from typing import Dict, List, Iterable, Tuple, Optional, Any


class Feature:
    def __init__(self, name, func):
        self.name = name
        self.func = func

    def __str__(self):
        return self.name

    def __call__(self, net):
        return self.func(net)


class ConstrainedFeature:
    def __init__(self, name, func, allowed):
        self.name = name
        self.func = func
        self.allowed = allowed

    def __str__(self):
        return self.name

    def __call__(self, net):
        return self.func(net, self.allowed)


def to_graph(edges):
    G = nx.DiGraph()
    for e_from, e_to in edges:
        G.add_edge(e_from, e_to)
    return G


def count_ops_opnodes(net: OpNodesNetGraph) -> Dict[int, int]:
    op_names, ops = net.ops, net.op_ids
    op_counts = {i: 0 for i, _ in enumerate(op_names)}
    for o in ops:
        op_counts[o] += 1

    return op_counts


def count_ops(net: NetGraph) -> Dict[int, int]:
    ops, edges = net.ops, net.edges
    op_counts = {i: 0 for i in ops.keys()}
    for val in edges.values():
        op_counts[val] += 1

    return op_counts


def get_in_out_edges_opnodes(net: OpNodesNetGraph, allowed: Iterable[int]):
    ops, graph = net.op_ids, net.graph
    in_edges: Dict[int, List[int]] = {i: [] for i, _ in enumerate(ops)}
    out_edges: Dict[int, List[int]] = {i: [] for i, _ in enumerate(ops)}

    for e in graph.edges:
        if ops[e[0]] in allowed:
            in_edges[e[1]].append(e[0])
        if ops[e[1]] in allowed:
            out_edges[e[0]].append(e[1])

    return in_edges, out_edges


def get_in_out_edges(net: NetGraph, allowed: Iterable[int]):
    G = net.to_graph()

    in_edges = {k: [e for e in G.edges if e[0] == k and net.edges[e] in allowed] for k in G.nodes}
    out_edges = {k: [e for e in G.edges if e[1] == k and net.edges[e] in allowed] for k in G.nodes}
    return in_edges, out_edges


def _min_path(G: nx.DiGraph, start: Any, end: Any, max_val: int) -> int:
    try:
        return nx.shortest_path_length(G, source=start, target=end)
    except (nx.NodeNotFound, nx.NetworkXNoPath):
        return end + 1 if max_val is None else max_val


def get_start_end(ops: List[int], start: int, end: int) -> Tuple[int, int]:
    res_start, res_end = None, None
    for i, o in enumerate(ops):
        if o == start:
            res_start = i
        elif o == end:
            res_end = i

    if res_start is None or res_end is None:
        raise ValueError(f"Start or end operation not found in the operations list: {ops}")

    return res_start, res_end


def is_valid_opnodes(op: int, allowed: Iterable[int], start: int = 0, end: int = 1) -> bool:
    if op == start or op == end or op in allowed:
        return True
    return False


def min_path_len_opnodes(net: OpNodesNetGraph, allowed: Iterable[int], start: int = 0,
                         end: int = 1, max_val: Optional[int] = None) -> int:
    ops, graph = net.op_ids, net.graph

    # filter out only allowed edges
    active_edges = []
    for e in graph.edges:
        if not is_valid_opnodes(ops[e[0]], allowed) or not is_valid_opnodes(ops[e[1]], allowed):
            continue
        active_edges.append(e)

    # get min path from start to end
    start, end = get_start_end(ops, start, end)
    return _min_path(to_graph(active_edges), start, end, max_val if max_val is not None else len(ops))


def min_path_len(net: NetGraph, allowed: List[Any], start: Any = 1, end: Any = 4, max_val: Optional[int] = None) -> int:
    edges = net.edges
    active_edges = {e for e, v in edges.items() if v in allowed}

    return _min_path(to_graph(active_edges), start, end, max_val if max_val is not None else len(edges))


def _max_num_path(G: nx.DiGraph, start: Any, end: Any, compute_weight: Callable) -> int:
    try:
        path = nx.shortest_path(G, source=start, target=end, weight=compute_weight, method='bellman-ford')
    except (nx.NodeNotFound, nx.NetworkXNoPath):
        return 0

    n_on_path = len(path)
    return n_on_path


def max_num_on_path_opnodes(net: OpNodesNetGraph, allowed: List[Any], start: Any = 0, end: Any = 1) -> int:
    ops, graph = net.op_ids, net.graph

    def is_allowed(node):
        node = ops[node]
        return node in allowed or node == start or node == end

    G_allowed = to_graph([edge for edge in graph.edges() if is_allowed(edge[0]) and is_allowed(edge[1])])

    def compute_weight(start, end, _):
        return -1

    start, end = get_start_end(ops, start, end)
    path_len = _max_num_path(G_allowed, start, end, compute_weight) - 2

    # number of nodes except input and output
    return path_len - 2


def max_num_on_path(net: NetGraph, allowed: List[Any], start: Any = 1, end: Any = 4) -> int:
    edges = net.edges
    
    def compute_weight(start, end, _):
        return -1

    path_len = _max_num_path(to_graph([k for k in edges.keys() if edges[k] in allowed]), start, end, compute_weight)

    # adjust by 1 -> edge count
    return path_len - 1
