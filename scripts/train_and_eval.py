import argparse
import os.path
import pickle
import time
from datetime import datetime

from sklearn.model_selection import train_test_split  # type: ignore
import wandb

import pandas as pd
from scipy.stats import spearmanr, kendalltau
from sklearn.ensemble import RandomForestRegressor  # type: ignore
from sklearn.metrics import r2_score, mean_squared_error  # type: ignore

from graf_nas.features.config import load_from_config
from graf_nas.graf import create_dataset, GRAF
from graf_nas.search_space import searchspace_classes, dataset_api_maps, DARTS
from naslib.utils import get_dataset_api  # type: ignore


zcps = ['epe_nas', 'fisher', 'flops', 'grad_norm', 'grasp', 'jacov', 'l2_norm', 'nwot', 'params', 'plain', 'snip', 'synflow', 'zen']


def get_train_test_splits(feature_dataset, y, train_size, seed):
    train_X, test_X, train_y, test_y = train_test_split(feature_dataset, y, train_size=train_size, random_state=seed)
    return {'train_X': train_X, 'train_y': train_y, 'test_X': test_X, 'test_y': test_y}


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

    assert args['debug_'] or args['wandb_key_'] is not None, "Need to provide a wandb key for syncing. To run without any logging, pass --debug_ to the script."
    # initialize wandb
    if not args['debug_']:
        cfg_args = {k: v for k, v in args.items() if not k.endswith('_')}
        wandb.login(key=args['wandb_key_'])
        wandb.init(project=args['wandb_project_'], config=cfg_args, name=f"{benchmark}_{dataset}_{get_timestamp()}")

    # get iterator of all available networks
    if benchmark == 'darts':
        df = pd.read_csv('../../zc_combine/data/nb301_nets.csv', index_col=0)
        net_iterator = (DARTS(n) for n in df.index)
    else:
        net_cls = get_searchspace_classes()[benchmark]
        dataset_api = get_dataset_api(search_space=dataset_api_maps[benchmark], dataset=dataset)
        net_iterator = net_cls.get_arch_iterator(dataset_api)

        if net_cls.random_iterator:
            raise ValueError("Not implemented for DARTS.")

    # load feature funcs and precomputed data
    feature_funcs = load_from_config(args['config'], benchmark)

    cached_data = None
    cached_zcp = [pd.read_csv(args['cached_zcp_path_'], index_col=0)] if args['cached_zcp_path_'] is not None else []
    cached_features = [pd.read_csv(args['cached_features_path_'], index_col=0)] if args['cached_features_path_'] is not None else []
    if len(cached_zcp) > 0 or len(cached_features) > 0:
        cached_data = pd.concat([*cached_zcp, *cached_features], axis=1)

    # load target data, compute features
    filename_args = ['benchmark', 'dataset', 'use_features', 'use_zcp', 'use_onehot']
    cache_path = f"{'_'.join([f'{fa}-{str(args[fa])}' for fa in filename_args])}_{os.path.splitext(os.path.basename(args['config']))[0]}.pickle"
    cache_path = f"{args['cache_prefix_']}_{cache_path}" if args['cache_prefix_'] is not None else cache_path
    if args['cache_dataset_'] and os.path.exists(cache_path):
        print(f"Loading cached dataset from {cache_path}.")
        with open(cache_path, 'rb') as f:
            cd = pickle.load(f)
            feature_dataset, y = cd['dataset'], cd['y']
    else:
        y = pd.read_csv(args['target_path_'], index_col=0)
        graf_model = GRAF(benchmark, features=feature_funcs, zcp_predictors=zcps, cached_data=cached_data, cache_features=True, no_zcp_raise=True)
        feature_dataset, y = create_dataset(graf_model, net_iterator, target_df=y, target_name=args['target_name'],
                                            use_features=args['use_features'], use_zcp=args['use_zcp'], use_onehot=args['use_onehot'],
                                            drop_unreachables='micro' in benchmark or benchmark == 'nb201')
        if args['cache_dataset_']:
            with open(cache_path, 'wb') as f:
                pickle.dump({'dataset': feature_dataset, 'y': y}, f)

    # get regressor
    model = RandomForestRegressor()

    # fit and eval N times
    for i in range(args['n_train_evals']):
        data_seed = args['seed'] + i
        data_splits = get_train_test_splits(feature_dataset, y, args['train_size'], data_seed)

        res = eval_model(model, data_splits)
        if not args['debug_']:
            wandb.log(res, step=data_seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run fit-eval of features/zero-cost proxies predictor with different sampling methods."
    )
    parser.add_argument('--benchmark', default='nb201', help="Which NAS benchmark to use (e.g. nb201).")
    parser.add_argument('--dataset', default='cifar10', help="Which dataset from the benchmark to use (e.g. cifar10).")
    parser.add_argument('--config', required=True, help="Path to the feature configuration file.")
    parser.add_argument('--cached_features_path_', default=None, help="Path to the cached features file.")
    parser.add_argument('--cached_zcp_path_', default=None, help="Path to the cached zcp score file.")
    parser.add_argument('--target_path_', required=True,
                        help="Path to network targets (e.g. accuracy). It should be a .csv file with net hashes as "
                             "index and `target_name` among the columns.")
    parser.add_argument('--target_name', default='val_accs', help="Name of the target column.")
    parser.add_argument('--seed', default=42,
                        help="Random seed for sampling the training data. For test data, `seed + 1` is used instead.")
    parser.add_argument('--n_train_evals', default=50, type=int,
                        help="Number of training samples on which the model is trained and evaluated.")
    parser.add_argument('--train_size', default=100, type=int,
                        help="Number of architectures to sample for the training set.")
    parser.add_argument('--wandb_key_', default=None, help='Login key to wandb.')
    parser.add_argument('--wandb_project_', default='graf', help="Wandb project name.")
    parser.add_argument('--debug_', action='store_true', help="If True, do not sync to wandb.")
    parser.add_argument('--use_features', action='store_true', help="If True, use features from GRAF.")
    parser.add_argument('--use_zcp', action='store_true', help="If True, use zero-cost proxies.")
    parser.add_argument('--use_onehot', action='store_true', help="If True, use the onehot encoding.")
    parser.add_argument('--cache_prefix_', default=None, help="Cache path filename prefix.")
    parser.add_argument('--cache_dataset_', action='store_true', help="If True, cache everything including zps.")

    args = parser.parse_args()
    args_dict = vars(args)
    train_end_evaluate(args_dict)
