import json
from functools import partial
from itertools import chain, combinations

from graf_nas.features.base import FeatureCallable


def load_from_config(func_cfg, func_dict):
    if isinstance(func_cfg, str):
        with open(func_cfg, 'r') as f:
            func_cfg = json.load(f)

    features = []
    for func_entry in func_cfg:
        func_list = load_function_group(func_entry, func_dict)
        features.extend(func_list)

    return features


def load_function_group(func_entry, func_key_dict):
    name = func_entry['name']
    func = func_key_dict[name]

    if 'allowed' in func_entry:
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
            func = partial(func, allowed=a)
            aname = f"{','.join(func_key_dict[a])}"
            res.append(FeatureCallable(f"{name}_{aname}", func))
        return res

    return [FeatureCallable(name, func)]


def get_op_combinations(op_list):
    return [c for c in chain.from_iterable(combinations(op_list, n) for n in range(len(op_list) + 1))]
