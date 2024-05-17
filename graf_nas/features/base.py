import networkx as nx


def to_graph(edges):
    G = nx.DiGraph()
    for e_from, e_to in edges:
        G.add_edge(e_from, e_to)
    return G


def _to_names(op_dict, ops):
    return {ops[o]: v for o, v in op_dict.items()}


def count_ops_opnodes(net):
    op_names, ops, _ = net
    op_counts = {i: 0 for i, _ in enumerate(op_names)}
    for o in ops:
        op_counts[o] += 1

    return op_counts


def count_ops(net):
    ops, edges = net
    op_counts = {i: 0 for i in ops.keys()}
    for val in edges.values():
        op_counts[val] += 1

    return op_counts


def get_in_out_edges_opnodes(net, allowed):
    _, ops, graph = net
    in_edges = {i: [] for i, _ in enumerate(ops)}
    out_edges = {i: [] for i, _ in enumerate(ops)}

    for e in graph.edges:
        if ops[e[0]] in allowed:
            in_edges[e[1]].append(e[0])
        if ops[e[1]] in allowed:
            out_edges[e[0]].append(e[1])

    return in_edges, out_edges


def get_in_out_edges(edges, allowed):
    G = to_graph(edges.keys())

    in_edges = {k: [e for e in G.edges if e[0] == k and edges[e] in allowed] for k in G.nodes}
    out_edges = {k: [e for e in G.edges if e[1] == k and edges[e] in allowed] for k in G.nodes}
    return in_edges, out_edges


def _min_path(edges, start, end, max_val):
    G = to_graph(edges)
    try:
        return nx.shortest_path_length(G, source=start, target=end)
    except (nx.NodeNotFound, nx.NetworkXNoPath):
        return end + 1 if max_val is None else max_val


def get_start_end(ops, start, end):
    res_start, res_end = None, None
    for i, o in enumerate(ops):
        if o == start:
            res_start = i
        elif o == end:
            res_end = i
    return res_start, res_end


def min_path_len_opnodes(net, banned, start=0, end=1, max_val=None):
    _, ops, graph = net

    active_edges = []
    for e in graph.edges:
        if ops[e[0]] in banned or ops[e[1]] in banned:
            continue
        active_edges.append(e)

    start, end = get_start_end(ops, start, end)
    return _min_path(active_edges, start, end, max_val)


def min_path_len(net, banned, start=1, end=4, max_val=None):
    _, edges = net
    active_edges = {e for e, v in edges.items() if v not in banned}

    return _min_path(active_edges, start, end, max_val)


def _max_num_path(G, start, end, compute_weight):
    try:
        path = nx.shortest_path(G, source=start, target=end, weight=compute_weight, method='bellman-ford')
    except (nx.NodeNotFound, nx.NetworkXNoPath):
        return 0

    n_on_path = 0
    for i, val in enumerate(path):
        if i == len(path) - 1:
            break
        n_on_path += (1 - compute_weight(val, path[i + 1], None)) // 2

    return n_on_path


def max_num_on_path_opnodes(net, allowed, start=0, end=1):
    _, ops, graph = net

    def compute_weight(start, end, _):
        return -1 if ops[end] in allowed else 1

    start, end = get_start_end(ops, start, end)

    return _max_num_path(graph, start, end, compute_weight)


def max_num_on_path(net, allowed, start=1, end=4):
    _, edges = net

    def compute_weight(start, end, _):
        return -1 if edges[(start, end)] in allowed else 1

    return _max_num_path(to_graph(edges.keys()), start, end, compute_weight)
