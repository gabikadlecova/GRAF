from sklearn.model_selection import train_test_split


def random_data_sample(feature_dataset, y, size, seed):
        x, _, y, _ = train_test_split(feature_dataset, y, train_size=size, random_state=seed)
        return x, y
