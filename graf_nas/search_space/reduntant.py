import networkx as nx

from zc_combine.features.conversions import to_graph


def remove_zero_branches(edges, zero_op=1, source=1, target=4):
    G = to_graph(edges)

    okay_edges = set()

    paths = nx.all_simple_paths(G, source=source, target=target)
    for path in map(nx.utils.pairwise, paths):
        path = [e for e in path]
        if any([edges[k] == zero_op for k in path]):
            continue

        for e in path:
            okay_edges.add(e)

    new_edges = {e: (val if e in okay_edges else zero_op) for e, val in edges.items()}
    return new_edges


def _remap_node(node, new, old):
    if node == old:
        return new
    return node


def remove_redundant_skips(edges, skip_op=0):
    G = to_graph(edges)

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

    return G, edges
