import argparse
import time
from datetime import datetime

import wandb

import pandas as pd
from scipy.stats import spearmanr, kendalltau
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

from graf_nas.features.config import load_from_config
from graf_nas.graf import create_dataset, GRAF
from graf_nas.search_space import searchspace_classes
from naslib.utils import get_dataset_api


zcps = ['epe_nas', 'fisher', 'flops', 'grad_norm', 'grasp', 'jacov', 'l2_norm', 'nwot', 'params', 'plain', 'snip', 'synflow', 'zen']


def get_train_test_splits(feature_dataset, y, train_size, test_size, seed):
    # TODO sampling strategies here
    raise NotImplementedError()


def eval_model(model, data_splits):
    start = time.time()
    model.fit(data_splits['train_X'], data_splits['train_y'])
    fit_time = time.time() - start

    preds = model.predict(data_splits['test_X'])
    true = data_splits['test_y']

    res = {
        'fit_time': fit_time,
        'r2': r2_score(true, preds),
        'mse': mean_squared_error(true, preds),
        'corr': spearmanr(preds, true)[0],
        'tau': kendalltau(preds, true)[0]
    }

    return res


def get_timestamp():
    return datetime.fromtimestamp(time.time()).strftime("%d-%m-%Y-%H-%M-%S-%f")


def train_end_evaluate(args):
    benchmark, dataset = args['benchmark'], args['dataset']

    # initialize wandb
    wandb.login(key=args['wandb_key_'])
    wandb.init(project=args['wandb_project_'], config=args, name=f"{benchmark}_{dataset}_{get_timestamp()}")

    # get iterator of all available networks
    net_cls = searchspace_classes[benchmark]
    dataset_api = get_dataset_api(search_space=benchmark, dataset=dataset)
    net_iterator = net_cls.get_arch_iterator(dataset_api)

    if net_cls.random_iterator:
        raise ValueError("Not implemented for DARTS.")

    feature_funcs = load_from_config(args['config'], benchmark)
    cached_zcp = pd.read_csv(args['cached_zcp_path_'], index_col=0)

    # load target data, compute features
    y = pd.read_csv(args['target_path_'], index_col=0)
    graf_model = GRAF(feature_funcs, benchmark, cached_data=cached_zcp, cache_features=False)
    feature_dataset = create_dataset(graf_model, net_iterator, zcps)

    # get regressor
    model = RandomForestRegressor()

    # fit and eval N times
    for i in range(args['n_train_evals']):
        data_seed = args['seed'] + i
        data_splits = get_train_test_splits(feature_dataset, y, args['train_size'], args['test_size'],
                                            data_seed)

        res = eval_model(model, data_splits)
        wandb.log(res, step=data_seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run fit-eval of features/zero-cost proxies predictor with different sampling methods."
    )
    parser.add_argument('--benchmark', default='nb201', help="Which NAS benchmark to use (e.g. nb201).")
    parser.add_argument('--dataset', default='cifar10', help="Which dataset from the benchmark to use (e.g. cifar10).")
    parser.add_argument('--config')
    parser.add_argument('--cached_zcp_path_')
    parser.add_argument('--target_path_')
    parser.add_argument('--seed')
    parser.add_argument('--n_train_evals')
    parser.add_argument('--wandb_key_')
    parser.add_argument('--wandb_project_', default='graf_sampling')

    args = parser.parse_args()
    args = vars(args)
    train_end_evaluate(args)
