import json
from itertools import chain, combinations

from collections.abc import Callable
from graf_nas.features import feature_dicts
from graf_nas.features.base import Feature, ConstrainedFeature
from graf_nas.search_space import searchspace_classes
from typing import Any, Iterable, List, Dict


def load_from_config(func_cfg: str | Dict, benchmark: str, config_loader_func: Callable | None = None) -> List[Callable]:
    """
    Load feature functions from a configuration json file (or a dict).

    :param func_cfg: path to the configuration file or a dictionary with the configuration
    :param benchmark_func: benchmark name to index `graf_nas.features.feature_dicts` and `graf_nas.search_space.searchspace_classes`
    :param config_loader_func: function that loads the configuration for a single function
        If None, graph_nas.features.config.load_function_group is used.
    :return: list of features
    """
    config_loader_func = config_loader_func if config_loader_func is not None else load_function_group

    if isinstance(func_cfg, str):
        with open(func_cfg, 'r') as f:
            func_cfg_dict = json.load(f)
    else:
        func_cfg_dict = func_cfg

    features: List[Callable] = []
    for func_entry in func_cfg_dict:
        func_list = config_loader_func(func_entry, benchmark)
        features.extend(func_list)

    return features


def load_function_group(func_entry: Dict, benchmark: str) -> List[Feature] | List[ConstrainedFeature]:
    """
    Load a function or a group of functions from a configuration entry. The group of functions is restricted to a
    set of allowed operations.
    
    A single function has the following structure:
    {
        "name": "function_name"
    }

    A group of functions has the following structure:
    {
        "name": "function_name",
        "allowed": ["op1", "op2", ...]
        "mode": "product" | "raw"  | "raw_str"  # optional, default - "product"
    }

    Depending on the mode, the allowed operations are interpreted as follows:
    - raw: each list entry represents one function with operations restricted to the given set
    - raw_str: same as raw, but each list entry is a comma-separated string of allowed operations
    - product (default): all possible combinations of the allowed operations are generated, each representing one function
        restricted to the given operation set

    Example:
    {
        "name": "function_name",
        "allowed": ["op1", "op3"],
        "mode": "product"
    }
    This will generate two "function_name": one with operations ["op1"], one with operations ["op2"],
    and one with operations ["op1", "op2"].

    :param func_entry: configuration entry
    :param benchmark: benchmark name to index `graf_nas.features.feature_dicts` and `graf_nas.search_space.searchspace_classes`
    :return: list with a single element if the function is not constrained, or a list of constrained functions
    """
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
                func_entry['allowed'] = [a.split(',') for a in func_entry['allowed']]
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


def get_op_combinations(op_list: List[Any]) -> List[Iterable[Any]]:
    """
    Get all possible subsets of a given operation set. The subsets are returned as a list of lists,
    ordered by the size of the subset.

    :param op_list: list of operations
    :return: list of all possible combinations of operations
    """
    return [c for c in chain.from_iterable(combinations(op_list, n) for n in range(1, len(op_list) + 1))]
