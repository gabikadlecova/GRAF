import pandas as pd
import torch

from abc import ABC, abstractmethod
from collections.abc import Callable

import torch.utils
import torch.utils.data
from graf_nas.search_space.base import NetBase
from naslib.predictors.zerocost import ZeroCost  # type: ignore
from naslib.utils import get_train_val_loaders  # type: ignore
from naslib.utils import load_config  # type: ignore
from naslib.predictors.utils.pruners.measures import available_measures  # type: ignore
from typing import Optional, Dict


class ZeroCostBase(ABC):
    """
    Abstract class for zero-cost proxy scorers.
    
    The __call__ method should accept a torch model and return a float value
    of the zero-cost proxy score.
    """
    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return self.name
    
    @abstractmethod
    def __call__(self, net: torch.nn.Module) -> float:
        pass


class ZeroCostNASLibProxy(ZeroCostBase):
    """
    Zero-cost proxy wrapper of the NASLib ZeroCost predictor.
    """
    def __init__(self, name: str, proxy: ZeroCost, dataloader: Optional[torch.utils.data.DataLoader] = None):
        super().__init__(name)
        self.proxy = proxy
        self.data_loader = dataloader

    def __call__(self, net: torch.nn.Module) -> float:
        return self.proxy.query(net, dataloader=self.data_loader)


def load_cached_zcp(net_hash: str, proxy_name: str, data_scores: pd.DataFrame):
    """
    Load a cached zero-cost proxy score from a pandas dataframe.

    :param net: network hash to index the dataframe
    :param proxy_name: zero-cost proxy column name
    :param data_scores: pandas dataframe with zero-cost proxy scores
    :return: zero-cost proxy score
    """
    assert isinstance(net_hash, str), "Net should be a string - network hash"
    assert proxy_name in data_scores.columns, f"Invalid proxy name: {proxy_name}, possible: {data_scores.columns}"

    return data_scores.loc[net_hash, proxy_name]


def get_zcp_predictor(proxy: str, dataloader: Optional[torch.utils.data.DataLoader] = None, **kwargs) -> ZeroCostBase:
    """
    Get a zero-cost proxy scorer from NASLib or from graf_nas.features.zero_cost.zero_cost_proxies.

    :param proxy: zero-cost proxy name
    :return: zero-cost proxy callable scorer
    """
    if proxy in available_measures:
        assert dataloader is not None, "Must provide dataloader if computing NASLib zero_cost scores."
        return ZeroCostNASLibProxy(proxy, ZeroCost(proxy, **kwargs), dataloader=dataloader)
    
    if proxy in zero_cost_proxies:
        return zero_cost_proxies[proxy](**kwargs)
    
    raise ValueError(f"Invalid zero-cost proxy: {proxy}, available: {available_measures}")


def get_zcp_dataloader(dataset: str, zc_cfg: str = '../zero_cost/NASLib/naslib/runners/zc/zc_config.yaml',
                       data: str = '../zero_cost/NASLib') -> torch.utils.data.DataLoader:
    """
    Get a dataloader for zero-cost proxy evaluation. See the NASLib documentation for more details.

    :param dataset: dataset name
    :param zc_cfg: path to zero-cost configuration file
    :param data: path to NASLib
    :return: dataloader for zero-cost proxy evaluation
    """
    cfg = load_config(zc_cfg)
    cfg.data = "{}/data".format(data)
    cfg.dataset = dataset
    loader, _, _, _, _ = get_train_val_loaders(cfg, mode='train')

    return loader


def parse_scores(zc_data, dataset: str, drop_constant: bool = True, nets_as_index: bool = True) -> pd.DataFrame:
    """
    Parse zero-cost scores from NAS-Bench-Suite Zero.
    The scores are save in json files. One file contains scores for all datasets in a benchmark.
    Each dataset is a dictionary of NASLib network hashes and a dictionary of evaluated zero-cost proxies
    (score and evaluation time in seconds) and validation accuracies.

    The result is a pandas dataframe with zero-cost scores as columns. Network hashes are used as the index,
    or if `nets_as_index` is False, saved as a column named 'net'. Validation accuracies are saved in the column
    'val_accs'.

    :param zc_data: input data loaded from json files
    :param dataset: dataset name to parse
    :param drop_constant: drop columns with constant values
    :param nets_as_index: set network hashes as output dataframe index
    :return: parsed scores as a pandas dataframe
    """
    assert dataset in zc_data, f"Invalid dataset: {dataset}, valid: {zc_data.keys()}"
    zc_data = zc_data[dataset]

    # get data of some random architecture
    arch_data = next(iter(zc_data))
    score_keys = [k for k in arch_data.keys() if k != 'id' and k != 'val_accuracy']

    data = []
    # get only val accs and scores without running time
    for net, arch_scores in zc_data.items():
        entry = {'net': net, 'val_accs': arch_scores['val_accuracy']}
        for s in score_keys:
            entry[s] = arch_scores[s]['score']

        data.append(entry)

    df = pd.DataFrame(data)
    if nets_as_index:
        df.set_index('net', inplace=True)

    # remove invalid (constant) score columns
    if drop_constant:
        drops = []
        for s in score_keys:
            if (df[s] == df.iloc[0][s]).all():
                drops.append(s)

        if len(drops):
            df.drop(columns=drops, inplace=True)

    return df


zero_cost_proxies: Dict[str, Callable] = {}
