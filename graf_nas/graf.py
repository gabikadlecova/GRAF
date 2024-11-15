import torch
from collections.abc import Iterable
from typing import Optional, Dict, Any, Hashable, List, Tuple

import time
import pandas as pd
import torch.utils
import torch.utils.data
from tqdm import tqdm

from graf_nas.features.config import load_from_config
from graf_nas.features.zero_cost import get_zcp_predictor, ZeroCostBase
from graf_nas.search_space.base import NetBase
from graf_nas.search_space.reduntant import remove_zero_branches


class GRAF:
    def __init__(self, benchmark: str, features=None, zcp_predictors: List[str | ZeroCostBase] | str | ZeroCostBase | None = None,
                 dataloader: Optional[torch.utils.data.DataLoader] = None,
                 cached_data: Optional[pd.DataFrame | Dict] = None, cache_zcp_scores: bool = True, cache_features: bool = True,
                 compute_new_zcps: bool = False, no_zcp_raise: bool = False, no_feature_raise: bool = False):

        # network benchmark (search space)
        self.benchmark = benchmark

        # features to compute - either as objects, dict config, or path to a config file
        self.features = features if features is not None else []
        if isinstance(features, str) or isinstance(features, dict):
            self.features = load_from_config(features, benchmark)  # load features from cfg file

        # zero-cost proxy predictors to use - either as objects or names
        self.zcp_predictors: Dict[str, ZeroCostBase | None] = {}
        zcp_predictors = zcp_predictors if zcp_predictors is not None else []
        zcp_predictors = [zcp_predictors] if isinstance(zcp_predictors, str) or isinstance(zcp_predictors, ZeroCostBase) else zcp_predictors
        for zcp in zcp_predictors:
            if isinstance(zcp, str):
                self.zcp_predictors[zcp] = None
            else:
                self.zcp_predictors[zcp.name] = zcp

        # for computing zero-cost proxies
        self.dataloader = dataloader
        self.compute_new_zcps = compute_new_zcps  # if False, return None if zcp is not found in cached data

        # cached data with precomputed features and/or zero-cost proxies
        self.cache_zcp_scores = cache_zcp_scores  # cache after computing zero-cost proxies
        self.cache_features = cache_features  # cache after computing features
        if cached_data is None:
            self.cached_data: Dict[str, Dict] | None = {} if (cache_zcp_scores or cache_features) else None
        elif isinstance(cached_data, pd.DataFrame):
            self.cached_data = {k: cached_data.loc[k].to_dict() for k in cached_data.index}
        else:
            assert isinstance(cached_data, dict), "Cached data must be a dictionary or a DataFrame."
            self.cached_data = cached_data

        # raise exceptions if a network has no precomputed score for a feature or a zero-cost proxy in the cached data
        self.no_zcp_raise = no_zcp_raise
        self.no_feature_raise = no_feature_raise

    def cached_data_to_df(self) -> pd.DataFrame:
        """
        Convert cached data to a pandas DataFrame.
        :return: pandas DataFrame with cached data
        """
        if self.cached_data is None:
            return pd.DataFrame()

        return pd.DataFrame(self.cached_data).T

    def compute_features(self, net: NetBase):
        """
        Compute all available features for a given network.
        :param net: network to compute features for
        :return: dictionary with computed features
        """
        net_graph = None
        res = {}

        # iterate over available features and compute/retrieve cached values
        for feat in self.features:
            def all_valid(f):
                if not isinstance(f, dict):
                    return True
                return all([(v is not None) for v in f.values()])

            # if everything already computed, retrieve cached features
            cached_feats = self.get_cached_feature(net.get_hash(), feat.name)
            if cached_feats is not None and all_valid(cached_feats):
                for k, v in cached_feats.items():
                    res[k] = v
                continue

            # optionally raise if not available
            if self.no_feature_raise:
                raise FeatureNotFoundException(f"Feature {feat.name} not found in precomputed data.")

            # otherwise compute, optionally cache
            net_graph = net_graph if net_graph is not None else net.to_graph()
            f_res = feat(net_graph)
            f_res = {feat.name: f_res} if not isinstance(f_res, dict) else {f"{feat.name}_{k}": v for k, v in f_res.items()}
            for fk, fv in f_res.items():
                res[fk] = fv
                if self.cache_features:
                    self._cache_score(net.get_hash(), fk, fv)

        return res

    def get_cached_feature(self, net: str, feat_name: str) -> Dict[Hashable, Any] | None:
        """
        Retrieve cached features for a given network hash.
        :param net: network hash
        :param feat_name: name of the feature
        :return: dictionary with cached features
        """
        if self.cached_data is None or net not in self.cached_data:
            return None

        # get all columns corresponding to one feature kind
        cached_entry = self.cached_data[net]
        colnames = [c for c in cached_entry.keys() if c.startswith(feat_name)]
        if len(colnames) == 0:
            return None

        return {k: cached_entry[k] for k in colnames}

    def get_cached_zcp(self, net: str, zcp_key: str) -> Any | None:
        """
        Retrieve cached zero-cost proxy score for a given network hash.
        :param net: network hash
        :param zcp_key: zero-cost proxy key
        :return: cached score or None if not found
        """
        # no caching or this zcp is not cached
        if self.cached_data is None or net not in self.cached_data:
            return None

        net_entry = self.cached_data[net]

        # return zcp data or None if not found
        return net_entry[zcp_key] if zcp_key in net_entry else None

    def compute_zcp(self, net: torch.nn.Module, zcp_name: str) -> float:
        """
        Compute zero-cost proxy score for a given network.
        :param net: network to compute zero-cost proxy for
        :param zcp_name: zero-cost proxy name
        :return: zero-cost proxy score
        """
        predictor = self._zcp_predictor(zcp_name)
        return predictor(net)

    def _zcp_predictor(self, name: str):
        """
        Get an initialized zero-cost proxy scorer or initialize it if not available.
        :param name: zero-cost proxy name
        :return: zero-cost proxy scorer
        """
        assert name in self.zcp_predictors, f"Zero-cost proxy {name} not available in the GRAF object."

        if self.zcp_predictors[name] is None:
            self.zcp_predictors[name] = get_zcp_predictor(name, dataloader=self.dataloader)

        return self.zcp_predictors[name]

    def _cache_score(self, net: str, colname: str, score: float):
        """
        Cache a score for a given network hash.
        :param net: network hash
        :param colname: column name to store the score
        :param score: score to cache
        """
        assert self.cached_data is not None, "No cached data available for storing scores."

        net_entry = self.cached_data.setdefault(net, {})
        net_entry[colname] = score

    def compute_zcp_scores(self, net: NetBase, return_times: bool = False) -> Dict[str, float | None]:
        """
        Compute zero-cost proxy scores for a given network.
        :param net: network to compute zero-cost proxies for
        :param zcp_names: list of zero-cost proxy names (or a single name)
        :return: dictionary with computed zero-cost proxy scores
        """
        # parse callable model
        res = {}
        times = {}
        for zcp_key in self.zcp_predictors.keys():
            # try to retrieve cached score
            result = self.get_cached_zcp(net.get_hash(), zcp_key)

            # compute score if not available or invalid; optionally cache it
            if self.compute_new_zcps and (result is None or pd.isnull(result)):
                # optionally raise if not available
                if self.no_zcp_raise:
                    raise FeatureNotFoundException(f"Zero-cost proxy {zcp_key} not found in precomputed data.")

                model = net.get_model()

                time_start = time.time()                
                result = self.compute_zcp(model, zcp_key)
                time_end = time_start - time.time()
                times[zcp_key] = time_end

                if self.cache_zcp_scores:
                    self._cache_score(net.get_hash(), zcp_key, result)

            res[zcp_key] = result

        return (res, times) if return_times else res


class FeatureNotFoundException(Exception):
    pass


def create_dataset(graf, nets: Iterable[NetBase], target_df: pd.DataFrame | None = None, target_name: str = 'val_accs',
                   drop_unreachables: bool = True, zero_op: int = 1, 
                   use_zcp: bool = False, use_features: bool = True, use_onehot: bool = False,
                   verbose: bool = True) -> pd.DataFrame | Tuple[pd.DataFrame, pd.Series]:
    """
    Create a dataset from a list of networks. Depending on the configuration (via `use_*` arguments), it computes features, zero-cost proxies,
    and one-hot encoding of the network graph.

    The result is a pandas dataframe with computed features, zero-cost proxies, and/or one-hot encoding. If provided, a series of corresponding
    target values is returned.

    Optionally, it can also drop networks with unreachable operations (e.g. due to zero-op in NAS-Bench-201) - i.e. duplicates in the search space.

    :param graf: GRAF object for computing features and zero-cost proxies
    :param nets: list of networks for which to compute features and zero-cost proxies
    :param target_df: dataframe with target values for each network (optional)
    :param target_name: name of the target column in `target_df`
    :param drop_unreachables: drop networks with unreachable operations (e.g. due to zero-op in NAS-Bench-201)
    :param zero_op: index of the zero operation for `drop_unreachables`
    :param use_zcp: if True, compute zero-cost proxies
    :param use_features: if True, compute features
    :param use_onehot: if True, compute one-hot encoding of the network graph
    :param verbose: if True, show progress bar
    :return: pandas dataframe with computed features and zero-cost proxies (and a series of corresponding target values if provided)
    """
    dataset = []
    y = []
    index = []
    for net in tqdm(nets, disable=not verbose):
        # discard networks with unreachable operations - these are not unique in the search space
        if drop_unreachables:
            graph = net.to_graph()
            new_graph = remove_zero_branches(graph, zero_op=zero_op)
            if new_graph.edges != graph.edges:
                continue

        # get the corresponding target value
        if target_df is not None:
            target = target_df.loc[net.get_hash()][target_name]
            y.append(target)

        # compute features for the network
        features = {}
        if use_features:
            features = graf.compute_features(net)

        # compute zero-cost proxies for the network
        if use_zcp:
            zcps = graf.compute_zcp_scores(net)
            features = {**features, **zcps}

        # compute one-hot encoding of the network graph
        if use_onehot:
            onehot = net.to_onehot()
            features = {**features, **{f"onehot_{i}": o for i, o in enumerate(onehot)}}

        # append the computed features to the dataset, and the network hash to the index
        index.append(net.get_hash())
        dataset.append(features)

    # create a dataset dataframe and return it with the target values (if provided)
    df = pd.DataFrame(dataset, index=index)
    if not len(y):
        return df

    return df, pd.Series(y, index=index)
