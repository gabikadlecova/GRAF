import click
import numpy as np
import pandas as pd
from tqdm import tqdm

from graf_nas import GRAF
from graf_nas.features.config import load_from_config
from graf_nas.features.zero_cost import get_zcp_dataloader


@click.command()
@click.option('--benchmark', default='nb201')
@click.option('--dataset', default='cifar10')
@click.option('--config', default='../graf_nas/configs/nb201.json')
@click.option('--cached_data', default=None)
@click.option('--naslib_root', default='../../zero_cost/NASLib')
@click.option('--zcp_cfg_path', default='../../zero_cost/NASLib/naslib/runners/zc/zc_config.yaml')
def main(benchmark, dataset, config, cached_data, naslib_root, zcp_cfg_path):
    feature_funcs = load_from_config(config, benchmark)

    kwargs = {}
    if zcp_cfg_path is not None:
        kwargs['zc_cfg'] = zcp_cfg_path
    if naslib_root is not None:
        kwargs['data'] = naslib_root

    dataloader = get_zcp_dataloader(dataset, **kwargs)
    cached_data = None if cached_data is None else pd.read_csv(cached_data)

    graf_model = GRAF(feature_funcs, benchmark, dataloader=dataloader, cached_data=None, compute_new_zcps=True,
                      cache_features=False)

    if benchmark == 'nb201':
        net = '(2, 3, 2, 3, 2, 3)'

        zcps = graf_model.compute_zcp_scores(net, ['synflow', 'nwot', 'l2_norm'])
        features = graf_model.compute_features(net)
    else:
        print('skipo')

    if 'macro' not in benchmark:
        ops = list(op_maps[benchmark]().keys())

    def rename_banned(c):
        if 'banned' in c:
            split_str = '_banned_'
            cond = lambda i, o: i not in o
        elif 'allowed' in c:
            split_str = '_allowed_'
            cond = lambda i, o: i in o
        else:
            return c

        c, old = c.split(split_str)
        old = old.split(')_')
        rest = '' if len(old) == 1 else f'_{old[1]}'
        old = old[0] if len(old) == 1 else f'{old[0]})'

        old = eval(old)
        old = [old] if not isinstance(old, tuple) else old
        ids = [o for i, o in enumerate(ops) if cond(i, old)]
        ids = [o for o in ids if o != 'input' and o != 'output']
        if benchmark == 'tnb101_micro':
            ids = [o for o in ids if 'avg' not in o]
        return f"{c}_({','.join(ids)}){rest}"

    cached_data.columns = [c.replace('[', '(').replace(']', ')') for c in cached_data.columns]
    cached_data.columns = [rename_banned(c) for c in cached_data.columns]

    for i in tqdm(cached_data.index):
        row = cached_data.loc[i]
        feats = graf_model.compute_features(row['net'])
        for f, val in feats.items():
            if f not in cached_data.columns:
                print(f"{f} not in cols!")

            if val != row[f]:
                graf_model.compute_features(row['net'])

            assert np.abs(val - row[f]) < np.finfo(float).eps, f"new: {val}, old: {row[f]}"


if __name__ == "__main__":
    main()
