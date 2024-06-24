# GRAF - Surprisingly Strong Performance Prediction with Neural Graph Features

Implementation of GRAF from our paper "Surprisingly Strong Performance Prediction with Neural Graph Features" ([paper](https://openreview.net/forum?id=EhPpZV6KLk)).

```
@inproceedings{kadlecova2024surprisingly,
title={Surprisingly Strong Performance Prediction with Neural Graph Features},
author={Gabriela Kadlecová and Jovita Lukasik and Martin Pilát and Petra Vidnerová and Mahmoud Safari and Roman Neruda and Frank Hutter},
booktitle={Forty-first International Conference on Machine Learning},
year={2024},
url={https://openreview.net/forum?id=EhPpZV6KLk}
}
```

**Note #1: Work in progress, will change during June/July '24 (you can follow us for updates!)**

**Note #2**: This is a refactored version of GRAF -- easier to use in your applications. For replicating
the results from the paper, refer to the repo [zc_combine](https://www.github.com/gabikadlecova/zc_combine)

## How to run

Clone this repo:
`git clone git@github.com:gabikadlecova/GRAF.git`

Additionally, clone `zc_combine` to access saved targets and ZCP (will be later included in this repo):

`git clone git@github.com:gabikadlecova/zc_combine.git`

Install GRAF:
```
cd GRAF
pip install -e .
```

Install [NASLib](https://www.github.com/automl/NASLib) in the Develop branch (might need a lower version of python) (TBD: fork where it works for newer python versions).

Run train and eval of the random forest predictor across 10 sample sizes:
```
cd scripts
python train_and_eval.py --benchmark nb201 --config ../graf_nas/configs/nb201.json \
    --cached_zcp_path_ ../../zc_combine/data/nb201_zc_proxies.csv \
    --target_path_ ../../zc_combine/data/nb201_val_accs.csv \
    --wandb_key_ <YOUR_WANDB_KEY> \
    --use_features --use_zcp \
    --train_size 100
```

Optionally, cache the features and zcps into a pickle file: `--cache_prefix_ test --cache_dataset_`

You can also run the training without any saved ZCP - they will be computed during the runtime of the script.


## To be implemented
- random iterator for darts/nb101
- cached ZCP and targets for all benchmarks
- better README
- how to add new GRAF or ZCPs