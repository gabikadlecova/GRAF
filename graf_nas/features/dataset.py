import pandas as pd
from itertools import chain, combinations


def get_feature_dataset(nets, config, func_dict, subset=None):
    dfs = {}
    for func_cfg in config:
        if subset is not None and func_cfg['name'] not in subset:
            continue
        dfs[func_cfg['name']] = load_from_config(nets, func_cfg, func_dict)
    return dfs


class FeatureCallable:
    def __init__(self, name, kwargs):
        self.name = name
        self.kwargs = kwargs  # not like this tho

# TODO config... allowed either raw or normal. raw is list copy paste, normal does the chain. It should be names.
# ... names would be converted to ids inside the funcs via op names
# TODO and this will then return a list of functions (maybe make it a class?)
def load_from_config(nets, func_cfg, func_dict):
    name = func_cfg['name']
    func = func_dict[name]
    if 'kwargs' not in func_cfg:
        return compute_feature_df(nets, func, name=name)

    if len(func_cfg['kwargs']) > 1:
        raise NotImplementedError("Combination of different kwarg settings not yet supported.")

    only_key = list(func_cfg['kwargs'].keys())[0]
    kwarg_list = [{only_key: val} for val in func_cfg['kwargs'][only_key]]
    func = eval_all_kwarg_settings(func, kwarg_list)
    return compute_feature_df(nets, func, name=name)


def _name_result(res, name):
    if isinstance(res, dict):
        return {f"{name}_{p}": v for p, v in res.items()}
    return {name: res}


#
def compute_feature_df(nets, feature_func, name=None):
    name = "" if name is None else f"{name}_"
    df = []
    for i, net in nets.items():
        res = {'idx': i}

        features = feature_func(net)
        if isinstance(features, list):
            for n, f in features:
                res.update(_name_result(f, f"{name}{n}"))
        elif isinstance(features, dict):
            res.update({f"{name}{k}": f for k, f in features.items()})
        else:
            res.update({name if len(name) else "value": features})

        df.append(res)

    df = pd.DataFrame(df)
    df.set_index('idx', inplace=True)
    return df


def eval_all_kwarg_settings(func, kwarg_list):
    def to_name(p):
        return '-'.join([f"{k}_{v}" for k, v in p.items()])

    def eval_func(net):
        return [(to_name(p), func(net, **p)) for p in kwarg_list]
    return eval_func


def get_op_combinations(op_list):
    return [c for c in chain.from_iterable(combinations(op_list, n) for n in range(len(op_list) + 1))]