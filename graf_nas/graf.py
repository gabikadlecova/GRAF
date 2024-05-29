import pandas as pd

from graf_nas.features import feature_dicts
from graf_nas.features.config import load_from_config


class GRAF:
    def __init__(self, features, benchmark, cached_zcp=None, cache_zcp_scores=True):
        self.features = features
        if isinstance(features, str) or isinstance(features, dict):
            self.features = load_from_config(features, feature_dicts[benchmark])  # load features from cfg file

        self.cached_zcp = cached_zcp
        if cached_zcp is None and cache_zcp_scores:
            self.cached_zcp = pd.DataFrame()

    def compute_features(self, net):
        res = {}
        for feat in self.features:
            f_res = feat(net)
            if isinstance(f_res, dict):
                for k, v in f_res.items():
                    res[f"{feat.name}_{k}"] = v
            else:
                res[feat.name] = f_res

        return res

    def get_cached_zcp(self, net):
        if self.cached_zcp is None:
            return None
        # TODO and else compute it

    def compute_zcps(self, net, zcp_names):
        if isinstance(zcp_names, str):
            zcp_names = [zcp_names]

        # TODO either get cached, or compute and optionally cache
