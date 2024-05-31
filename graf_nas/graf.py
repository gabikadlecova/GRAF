import numpy as np
import pandas as pd
import torch

from graf_nas.features import feature_dicts
from graf_nas.search_space.conversions import convert_to_naslib, encode_to_graph
from graf_nas.features.config import load_from_config
from graf_nas.features.zero_cost import get_zcp_predictor


class GRAF:
    def __init__(self, features, benchmark, dataloader=None, cached_data=None, cache_zcp_scores=True,
                 cache_features=True, compute_new_zcps=False):

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

    def compute_features(self, net):
        net_graf = None
        res = {}
        for feat in self.features:
            def feats_not_none(f):
                if f is None:
                    return False
                if not isinstance(f, dict):
                    return True
                return all([(v is not None) for v in f.values()])

            # if already computed, retrieve cached features
            cached_feats = self.get_cached_feature(net, feat.name)
            if feats_not_none(cached_feats):
                for k, v in cached_feats.items():
                    res[k] = v
                continue

            # otherwise compute and optionally cache
            net_graf = net_graf if net_graf is not None else encode_to_graph(net, self.benchmark)
            f_res = feat(net_graf)
            f_res = {feat.name: f_res} if not isinstance(f_res, dict) else {f"{feat.name}_{k}": v for k, v in f_res.items()}
            for fk, fv in f_res.items():
                res[fk] = fv
                if self.cache_features:
                    self._cache_score(net, fk, fv)

        return res

    def get_cached_feature(self, net, feat_name):
        if self.cached_data is None or net not in self.cached_data.index:
            return None

        colnames = [c for c in self.cached_data.columns if c.startswith(feat_name)]
        if not len(colnames):
            return None

        return {k: self.cached_data.loc[net][k] for k in colnames}

    def get_cached_zcp(self, net, zcp_key):
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

    def _cache_score(self, net, colname, score):
        if colname not in self.cached_data.columns:
            self.cached_data[colname] = None

        if net not in self.cached_data.index:
            self.cached_data.loc[net] = {c: None for c in self.cached_data.columns}

        self.cached_data.loc[net, colname] = score

    def compute_zcp_scores(self, net, zcp_names):
        if isinstance(zcp_names, str):
            zcp_names = [zcp_names]

        naslib_net = None
        res = {}
        for zcp_key in zcp_names:
            # try to retrieve cached score
            result = self.get_cached_zcp(net, zcp_key)

            # compute score if not available or invalid; optionally cache it
            if self.compute_new_zcps and (result is None or pd.isnull(result)):
                if naslib_net is None:
                    naslib_net = convert_to_naslib(net, self.benchmark)
                    naslib_net.parse()

                result = self.compute_zcp(naslib_net, zcp_key)
                if self.cache_zcp_scores:
                    self._cache_score(net, zcp_key, result)

            res[zcp_key] = result

        return res
