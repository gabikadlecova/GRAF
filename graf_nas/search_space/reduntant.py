import networkx as nx  # type: ignore

from graf_nas.search_space.conversions import NetGraph


def remove_zero_branches(net: NetGraph, zero_op: int = 1, source: int = 1, target: int = 4) -> NetGraph:
    """
    Removes zero branches from the network.
    
    Due to the zero operations, some branches may have no effect on the output, or
    may not get any information from the input.

    This function zeros out the edges that are not part of any non-zero path from the source to the target.
    :param net: network graph
    :param zero_op: the zero operation id (default 1 in NASLib NB201 encoding)
    :param source: source node id
    :param target: target node id
    :return: network graph with zero branches removed
    """
    G = net.to_graph()

    okay_edges = set()

    paths = nx.all_simple_paths(G, source=source, target=target)
    for path in map(nx.utils.pairwise, paths):
        path = [e for e in path]
        if any([net.edges[k] == zero_op for k in path]):
            continue

        for e in path:
            okay_edges.add(e)

    new_edges = {e: (val if e in okay_edges else zero_op) for e, val in net.edges.items()}
    return NetGraph(net.ops, new_edges, cache_graph=net.cache_graph)


def _remap_node(node, new, old):
    if node == old:
        return new
    return node


def remove_redundant_skips(net: NetGraph, skip_op: int = 0, zero_op: int = 1) -> NetGraph:
    """
    Removes redundant skip connections from the network.
    Checks if the skip connection is redundant by removing it and checking if there is still a path between the nodes.
    If yes, the skip connection is needed, otherwise it is removed.

    :param net: network graph
    :param skip_op: the skip connection operation id (default 0 in NASLib NB201 encoding)
    :param zero_op: the zero operation id (default 1 in NASLib NB201 encoding)
    :return: network graph with redundant skip connections removed
    """
    G = net.to_graph()
    # look only at nonzero ops
    edges = {k: v for k, v in net.edges.items() if v != zero_op}

    while True:
        e_removed = False
        e_keys = edges.keys()

        for e in e_keys:
            if edges[e] == skip_op and e in G.edges:
                G.remove_edge(*e)
                u, v = e
                try:
                    nx.shortest_path(G, source=u, target=v)
                    # return edge if there is still some path after removing the skip - not reduntant
                    G.add_edge(*e)
                except nx.NetworkXNoPath:
                    # can remove skip - redundant, contract nodes
                    G = nx.contracted_nodes(G, u, v)
                    edges.pop((u, v))
                    edges = {(_remap_node(k[0], u, v), _remap_node(k[1], u, v)): val for k, val in edges.items()}
                    e_removed = True
                    break

        if not e_removed:
            break

    return NetGraph(net.ops, edges, cache_graph=net.cache_graph)
