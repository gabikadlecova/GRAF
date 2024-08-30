from collections.abc import Iterable
from typing import Union

import pandas as pd
from tqdm import tqdm

from graf_nas.features import feature_dicts
from graf_nas.features.config import load_from_config
from graf_nas.features.zero_cost import get_zcp_predictor
from graf_nas.search_space.conversions import NetBase
from graf_nas.search_space.reduntant import remove_zero_branches


class GRAF:
    def __init__(self, features, benchmark, dataloader=None, cached_data=None, cache_zcp_scores=True,
                 cache_features=True, compute_new_zcps=False, no_zcp_raise=False, no_feature_raise=False):

        self.benchmark = benchmark
        self.features = features
        if isinstance(features, str) or isinstance(features, dict):
            self.features = load_from_config(features, feature_dicts[benchmark])  # load features from cfg file

        self.cached_data = cached_data
        self.cache_zcp_scores = cache_zcp_scores
        self.cache_features = cache_features
        if cached_data is None and (cache_zcp_scores or cache_features):
            self.cached_data = pd.DataFrame()

        self.zcp_predictors = {}
        self.dataloader = dataloader
        self.compute_new_zcps = compute_new_zcps

        self.no_zcp_raise = no_zcp_raise
        self.no_feature_raise = no_feature_raise

    def compute_features(self, net: NetBase):
        net_graph = None
        res = {}
        for feat in self.features:
            def feats_not_none(f):
                if f is None:
                    return False
                if not isinstance(f, dict):
                    return True
                return all([(v is not None) for v in f.values()])

            # if already computed, retrieve cached features
            cached_feats = self.get_cached_feature(net.get_hash(), feat.name)
            if feats_not_none(cached_feats):
                for k, v in cached_feats.items():
                    res[k] = v
                continue

            # optionally raise if not available
            if self.no_feature_raise:
                raise FeatureNotFoundException(f"Feature {feat.name} not found in precomputed data.")

            # otherwise compute and optionally cache
            net_graph = net_graph if net_graph is not None else net.to_graph()
            f_res = feat(net_graph)
            f_res = {feat.name: f_res} if not isinstance(f_res, dict) else {f"{feat.name}_{k}": v for k, v in f_res.items()}
            for fk, fv in f_res.items():
                res[fk] = fv
                if self.cache_features:
                    self._cache_score(net.get_hash(), fk, fv)

        return res

    def get_cached_feature(self, net: str, feat_name):
        if self.cached_data is None or net not in self.cached_data.index:
            return None

        colnames = [c for c in self.cached_data.columns if c.startswith(feat_name)]
        if not len(colnames):
            return None

        features = self.cached_data.loc[net, colnames]
        return features.to_dict()

    def get_cached_zcp(self, net: str, zcp_key):
        # no caching or this zcp is not cached
        if self.cached_data is None or zcp_key not in self.cached_data.columns:
            return None

        # no zcp data for this net
        if net not in self.cached_data.index:
            return None

        return self.cached_data.loc[net][zcp_key]

    def compute_zcp(self, net, zcp_name):
        assert self.dataloader is not None, "Must provide dataloader if computing zero_cost scores."
        pred = self._zcp_predictor(zcp_name)
        result = pred.query(net, dataloader=self.dataloader)

        return result

    def _zcp_predictor(self, name):
        if name not in self.zcp_predictors:
            self.zcp_predictors[name] = get_zcp_predictor(name)
        return self.zcp_predictors[name]

    def _cache_score(self, net: str, colname, score):
        if colname not in self.cached_data.columns:
            self.cached_data[colname] = None

        if net not in self.cached_data.index:
            self.cached_data.loc[net] = {c: None for c in self.cached_data.columns}

        self.cached_data.loc[net, colname] = score

    def compute_zcp_scores(self, net: NetBase, zcp_names):
        if isinstance(zcp_names, str):
            zcp_names = [zcp_names]

        model = None
        res = {}
        for zcp_key in zcp_names:
            # try to retrieve cached score
            result = self.get_cached_zcp(net.get_hash(), zcp_key)

            # compute score if not available or invalid; optionally cache it
            if self.compute_new_zcps and (result is None or pd.isnull(result)):
                # optionally raise if not available
                if self.no_zcp_raise:
                    raise FeatureNotFoundException(f"Zero-cost proxy {zcp_key} not found in precomputed data.")

                # parse callable model
                if model is None:
                    model = net.get_model()

                result = self.compute_zcp(model, zcp_key)
                if self.cache_zcp_scores:
                    self._cache_score(net.get_hash(), zcp_key, result)

            res[zcp_key] = result

        return res


class FeatureNotFoundException(Exception):
    pass


def create_dataset(graf, nets: Iterable[NetBase], target_df=None, zcp_names: list[str] | None = None, drop_unreachables=True,
                   zero_op=1, target_name='val_accs', use_zcp=False, use_features=True, use_onehot=False, verbose=True):
    dataset = []
    y = []
    index = []
    for net in tqdm(nets, disable=not verbose):
        if drop_unreachables:
            _, edges = net.to_graph()
            new_edges = remove_zero_branches(edges, zero_op=zero_op)
            if new_edges != edges:
                continue

        if target_df is not None:
            target = target_df.loc[net.get_hash()][target_name]
            y.append(target)

        features = {}
        if use_features:
            features = graf.compute_features(net)

        if use_zcp:
            assert zcp_names is not None
            zcps = graf.compute_zcp_scores(net, zcp_names)
            features = {**features, **zcps}

        if use_onehot:
            onehot = net.to_onehot()
            features = {**features, **{f"onehot_{i}": o for i, o in enumerate(onehot)}}

        index.append(net.get_hash())
        dataset.append(features)

    dataset = pd.DataFrame(dataset, index=index)
    if not len(y):
        return dataset

    return dataset, pd.Series(y, index=index)
