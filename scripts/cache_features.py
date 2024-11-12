import click
import pandas as pd
from tqdm import tqdm

from graf_nas import GRAF
from graf_nas.features.config import load_from_config
from graf_nas.graf import create_dataset
from graf_nas.search_space import get_searchspace_classes, dataset_api_maps, DARTS
from naslib.utils import get_dataset_api


@click.command()
@click.option('--benchmark', default='nb201')
@click.option('--dataset', default='cifar10', help='Required only for the dataset api.')
@click.option('--config', default='../graf_nas/configs/nb201.json')
@click.option('--out_path', required=True)
def main(benchmark, dataset, config, out_path):
    feature_funcs = load_from_config(config, benchmark)

    graf_model = GRAF(feature_funcs, benchmark, cache_features=False)

    net_cls = get_searchspace_classes()[benchmark]

    if benchmark == 'darts':
        df = pd.read_csv('../../zc_combine/data/nb301_nets.csv', index_col=0)
        net_iterator = (DARTS(n) for n in df.index)
    else:
        dataset_api = get_dataset_api(search_space=dataset_api_maps[benchmark], dataset=dataset)
        net_iterator = net_cls.get_arch_iterator(dataset_api)

        if net_cls.random_iterator:
            raise ValueError("Not implemented for DARTS.")

    feature_dataset = create_dataset(graf_model, net_iterator, use_features=True,
                                     drop_unreachables='micro' in benchmark or benchmark == 'nb201')
    feature_dataset.to_csv(out_path)


if __name__ == "__main__":
    main()
