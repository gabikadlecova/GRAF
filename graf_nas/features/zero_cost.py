import pandas as pd
from naslib.predictors.zerocost import ZeroCost
from naslib.utils import get_train_val_loaders
from naslib.utils import load_config


def load_cached_zcp(net, proxy_name, data_scores):
    assert isinstance(net, str), "Net should be a string - network hash"
    assert proxy_name in data_scores.columns, f"Invalid proxy name: {proxy_name}, possible: {data_scores.columns}"
    row = data_scores.loc[net]
    return row[proxy_name]


def get_zcp_predictor(proxy):
    return ZeroCost(proxy)


def get_proxy_dataloader(dataset, zc_cfg='../zero_cost/NASLib/naslib/runners/zc/zc_config.yaml',\
                         data='../zero_cost/NASLib'):
    cfg = load_config(zc_cfg) # tak nejak, je to v utils
    cfg.data = "{}/data".format(data)
    cfg.dataset = dataset
    loader, _, _, _ = get_train_val_loaders(cfg, mode='train')

    return loader


def compute_zcp(net, proxy, loader):
    net.parse()
    return proxy.query(net, dataloader=loader)


def parse_scores(zc_data, dataset, drop_constant=True, nets_as_index=True):
    assert dataset in zc_data, f"Invalid dataset: {dataset}, valid: {zc_data.keys()}"
    zc_data = zc_data[dataset]

    # get data of some random architecture
    arch_data = next(iter(zc_data))
    score_keys = [k for k in arch_data.keys() if k != 'id' and k != 'val_accuracy']

    df = []
    # get only val accs and scores without running time
    for net, arch_scores in zc_data.items():
        entry = {'net': net, 'val_accs': arch_scores['val_accuracy']}
        for s in score_keys:
            entry[s] = arch_scores[s]['score']

        df.append(entry)

    df = pd.DataFrame(df)
    if nets_as_index:
        df.set_index('net', inplace=True)

    # remove invalid (constant) scores
    if drop_constant:
        drops = []
        for s in score_keys:
            if (df[s] == df.iloc[0][s]).all():
                drops.append(s)

        if len(drops):
            df.drop(columns=drops, inplace=True)

    return df
