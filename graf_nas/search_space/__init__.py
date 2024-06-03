from graf_nas.search_space.darts import DARTS
from graf_nas.search_space.nasbench101 import NB101
from graf_nas.search_space.nasbench201 import NB201
from graf_nas.search_space.tnb101_macro import TNB101_macro
from graf_nas.search_space.tnb101_micro import TNB101_micro

searchspace_classes = {
    'nb101': NB101,
    'nb201': NB201,
    'darts': DARTS,
    'tnb101_micro': TNB101_micro,
    'tnb101_macro': TNB101_macro
}
