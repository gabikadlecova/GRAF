import json
from itertools import chain, combinations

from graf_nas.features import feature_dicts
from graf_nas.features.base import Feature, ConstrainedFeature
from graf_nas.search_space import searchspace_classes


def load_from_config(func_cfg, func_dict):
    if isinstance(func_cfg, str):
        with open(func_cfg, 'r') as f:
            func_cfg = json.load(f)

    features = []
    for func_entry in func_cfg:
        func_list = load_function_group(func_entry, func_dict)
        features.extend(func_list)

    return features


def load_function_group(func_entry, benchmark):
    name = func_entry['name']
    func_key_dict = feature_dicts[benchmark]
    func = func_key_dict[name]
    bench_op_map = None

    if 'allowed' in func_entry:
        if bench_op_map is None:
            searchspace_cls = searchspace_classes[benchmark]
            if not hasattr(searchspace_cls, 'get_op_map'):
                raise ValueError(f"Searchspace {benchmark} has no method for op map "
                                 f"(needed when 'allowed' is in the config file).")
            bench_op_map = searchspace_cls.get_op_map()

        is_raw = False
        if 'allowed_mode' in func_entry:
            allowed_mode = func_entry['allowed_mode']
            if allowed_mode == 'raw':
                is_raw = True
            elif allowed_mode == 'raw_str':
                func_entry['allowed'] = func_entry['allowed'].split(',')
            elif allowed_mode == 'product':
                is_raw = False
            else:
                raise ValueError(f'mode must be one of ["raw", "raw_str", "product"], but got {allowed_mode}')

        allowed = func_entry['allowed'] if is_raw else get_op_combinations(func_entry['allowed'])

        res = []
        for a in allowed:
            aname = f"({','.join(a)})"
            a = [bench_op_map[op] for op in a]
            res.append(ConstrainedFeature(f"{name}_{aname}", func, a))
        return res

    return [Feature(name, func)]


def get_op_combinations(op_list):
    return [c for c in chain.from_iterable(combinations(op_list, n) for n in range(1, len(op_list) + 1))]
