from graf_nas.search_space.darts import darts_to_graph
from graf_nas.search_space.nasbench101 import nb101_to_graph, nb101_to_onehot
from graf_nas.search_space.nasbench201 import nb201_to_graph, tnb101_to_graph
from graf_nas.search_space.tnb101_macro import tnb101_macro_encode

from naslib.search_spaces.nasbench201.encodings import encode_adjacency_one_hot_op_indices
from naslib.search_spaces.nasbench301.encodings import encode_adj, encode_gcn
from naslib.search_spaces.transbench101.encodings import encode_adjacency_one_hot_transbench_micro_op_indices
from naslib.search_spaces.transbench101.encodings import encode_adjacency_one_hot_transbench_macro_op_indices

from naslib.search_spaces.nasbench101.graph import NasBench101SearchSpace
from naslib.search_spaces.nasbench201.graph import NasBench201SearchSpace
from naslib.search_spaces.nasbench301.graph import NasBench301SearchSpace
from naslib.search_spaces.transbench101.graph import TransBench101SearchSpaceMicro, TransBench101SearchSpaceMacro


def encode_to_graph(net, benchmark):
    return bench_conversions[benchmark](net)


def encode_to_onehot(net, benchmark):
    return onehot_conversions[benchmark](net)


def convert_to_naslib(net, benchmark, **kwargs):
    if isinstance(net, str):
        net = eval(net)

    naslib_obj = naslib_objects[benchmark](**kwargs)
    naslib_obj.set_spec(net)
    return naslib_obj


bench_conversions = {
    'nb101': nb101_to_graph,
    'nb201': nb201_to_graph,
    'darts': darts_to_graph,
    'tnb101_micro': tnb101_to_graph,
    'tnb101_macro': tnb101_macro_encode
}


onehot_conversions = {
    'nb101': nb101_to_onehot,
    'nb201': encode_adjacency_one_hot_op_indices,
    'darts': encode_adj,
    'tnb101_micro': encode_adjacency_one_hot_transbench_micro_op_indices,
    'tnb101_macro': encode_adjacency_one_hot_transbench_macro_op_indices
}


naslib_objects = {
    'nb101': NasBench101SearchSpace,
    'nb201': NasBench201SearchSpace,
    'tnb101_micro': TransBench101SearchSpaceMicro,
    'darts': NasBench301SearchSpace,
    'tnb101_macro': TransBench101SearchSpaceMacro
}
