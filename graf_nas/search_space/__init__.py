def get_searchspace_classes():
    from graf_nas.search_space.darts import DARTS
    from graf_nas.search_space.nasbench101 import NB101
    from graf_nas.search_space.nasbench201 import NB201
    from graf_nas.search_space.tnb101_macro import TNB101_macro
    from graf_nas.search_space.tnb101_micro import TNB101_micro

    # Maps search space names to their classes
    return {
        'nb101': NB101,
        'nb201': NB201,
        'darts': DARTS,
        'tnb101_micro': TNB101_micro,
        'tnb101_macro': TNB101_macro
    }


def get_dataset_apis():
    # Maps graf_nas names to NASLib search space names
    return {
        'nb101': 'nasbench101',
        'nb201': 'nasbench201',
        'darts': 'nasbench301',
        'tnb101_macro': 'transbench101_macro',
        'tnb101_micro': 'transbench101_micro',
    }
