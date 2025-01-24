"""
  Created on Jan 2025
@author: Elena Markova
          for Attrition Rate Project
"""

import pickle
from pandas import read_csv
from tensorflow import keras as K
import pandas as pd

from utils.custom_metric import calc_metric

def test_model(_model: K.Model, _feat: pd.DataFrame, _trg: pd.DataFrame):
    pred = model.predict(feat)
    ground_truth = []
    for status in _trg.status:
        ground_truth.append(status)
    calc_metric(pred, ground_truth, 0.8)



if __name__ == '__main__':
    config = {
        "test_data_path": "data/dataset-nn-small.csv",
        "model_path": "model.pkl"
    }

    model = pickle.load(open(config["model_path"], 'rb'))
    dataset = read_csv(config["test_data_path"]).transpose()
    feat = dataset[:-1].transpose()
    trg = dataset[-1:].transpose()

    test_model(model, feat, trg)