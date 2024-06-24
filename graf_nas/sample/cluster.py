import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
import wandb
from sklearn.decomposition import PCA

cluster_methods = {
    'DBSCAN': DBSCAN,
    'KMeans': KMeans,
    'hier': AgglomerativeClustering
}


def get_data_clusters(feature_df, cluster_method, **kwargs):
    cluster_estimator = cluster_methods[cluster_method](**kwargs)
    cluster_estimator.fit(feature_df)
    return pd.Series(cluster_estimator.labels_, name='labels', index=feature_df.index)


def clustered_data_sample(feature_dataset, y, size, seed, weighted=False, replace=False, cluster_method='DBSCAN', **kwargs):
    labels = get_data_clusters(feature_dataset, cluster_method, **kwargs)
    rng = np.random.RandomState(seed)

    uniques, counts = np.unique(labels, return_counts=True)
    pb = counts / len(labels)

    wandb.log({'n_clusters': len(uniques)}, step=seed)

    # select with replacement or without (when sure we don't run out of examples)
    if replace or len(uniques) >= size:
        clusters = rng.choice(uniques, size=size, replace=replace, p=pb if weighted else None)
    else:
        # each cluster should appear the same number of times (off by one for the last iteration)
        clusters = []
        selected = 0
        np.random.shuffle(uniques)
        while len(clusters) < size:
            if selected == len(uniques):
                selected = 0
                np.random.shuffle(uniques)

            clusters.append(uniques[selected])
            selected += 1
        clusters = np.array(clusters)

    # sample one net from each cluster
    dataset_idx = []
    for c in clusters:
        example_net = np.random.choice(labels[labels == c].index, size=1)[0]
        dataset_idx.append(example_net)

    return feature_dataset.loc[dataset_idx], y.loc[dataset_idx]
