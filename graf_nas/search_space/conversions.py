from abc import ABC, abstractmethod


class NetBase(ABC):
    def __init__(self, net):
        self.net = net

    def get_hash(self):
        return self.net

    @abstractmethod
    def to_graph(self):
        raise NotImplementedError()

    @abstractmethod
    def to_onehot(self):
        raise NotImplementedError()

    @abstractmethod
    def to_naslib(self):
        raise NotImplementedError()


def convert_to_naslib(net, naslib_object, **kwargs):
    if isinstance(net, str):
        net = eval(net)

    naslib_obj = naslib_object(**kwargs)
    naslib_obj.set_spec(net)
    return naslib_obj
