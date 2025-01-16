"""soup: A Flower / sklearn app."""

import numpy as np
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

fds = None  # Cache FederatedDataset
data_path = 'data/dataset-2.csv'

def get_model(penalty: str, local_epochs: int):
    print("Getting model")
    model = RandomForestRegressor(
        n_estimators=10,
        max_features='sqrt',
        # max_iter=local_epochs,
        warm_start=True,
    )
    print("Initial params after model creation:", model.get_params())
    return model


def set_initial_params(model):
    model.estimators_ = []


def get_model_params(model):
    params = [model.estimators_]
    return params