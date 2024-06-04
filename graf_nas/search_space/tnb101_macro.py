from graf_nas.search_space.conversions import convert_to_naslib, NetBase

from naslib.search_spaces.transbench101.encodings import encode_adjacency_one_hot_transbench_macro_op_indices
from naslib.search_spaces.transbench101.graph import TransBench101SearchSpaceMacro


class TNB101_macro(NetBase):
    naslib_object = None
    random_iterator = False

    def __init__(self, net):
        super().__init__(net)

    def to_graph(self):
        return tnb101_macro_encode(self.net)

    def to_onehot(self):
        return encode_adjacency_one_hot_transbench_macro_op_indices(eval(self.net))

    def to_naslib(self):
        return convert_to_naslib(self.net, TransBench101SearchSpaceMacro)

    @staticmethod
    def get_arch_iterator(dataset_api):
        if TNB101_macro.naslib_object is None:
            TNB101_macro.naslib_object = TransBench101SearchSpaceMacro()

        for n in TNB101_macro.naslib_object.get_arch_iterator(dataset_api):
            yield TNB101_macro(str(n))


def tnb101_macro_encode(net):
    if isinstance(net, str):
        net = net.strip('()').split(',')

    res = []

    for idx in net:
        encoding = {}
        idx = int(idx)
        encoding['channel'] = idx % 2 == 0
        encoding['stride'] = idx > 2
        res.append(encoding)

    return res
