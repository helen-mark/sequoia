"""
  Created on Jan 2025
@author: Elena Markova
          for Attrition Rate Project
"""

import auton_survival
import pandas as pd
from auton_survival import estimators
from auton_survival.preprocessing import Preprocessor
from pandas import read_csv

from utils.custom_metric import calc_metric


def preprocess(_features: pd.DataFrame):
    return Preprocessor().fit_transform(_features,
                                            cat_feats=['department', 'citizenship', 'gender', 'family_status'],
                                            num_feats=[
                                                        "age",
                                                        "vacation_days_shortterm",
                                                        "vacation_days_longterm",
                                                        "income_shortterm",
                                                        "income_longterm",
                                                        "overtime_shortterm",
                                                        "overtime_longterm",
                                                        "absenteeism_shortterm",
                                                        "absenteeism_longterm"
                                                        # "external_factor_1",
                                                        # "external_factor_2",
                                                        # "external_factor_3"
                                                        ]
                                            )


def fit_survival_machine(_features: pd.DataFrame, _outcomes: pd.DataFrame):
    model = estimators.SurvivalModel(model='dsm')
    # Preprocessor does both imputing and scaling of data:
    _features = preprocess(_features)

    print("Fitting...")
    model.fit(_features, _outcomes)

    # metrics = ['brs', 'ibs', 'auc', 'ctd']
    # score = survival_regression_metric(metric='brs', _outcomes,
    #                                    _outcomes, predictions,
    #                                    times=[20])

    return model



def test_survival_machine(_model: auton_survival.models.dsm.DeepSurvivalMachines, _feats, _outs):
    feats = preprocess(_feats)
    times = []
    true_events = []
    predictions = []
    for t in _outs.time:
        times.append(t)
    for e in _outs.event:
        true_events.append(e)
    for n, f in enumerate(feats.transpose()):
        pred = _model.predict_risk(feats.iloc[[n]], times=[times[n]])  # predict risk of event at ground-truth event time
        predictions.append(pred[0][0])

    calc_metric(predictions, true_events, 0.5)



def prepare_dataset(_data_path: str):
    dataset = read_csv(_data_path, delimiter=',')
    val_dataframe = dataset.sample(frac=0.2, random_state=133)
    train_dataframe = dataset.drop(val_dataframe.index)

    dataset_t = train_dataframe.transpose()
    feats_t = dataset_t[:-2].transpose()
    outs_t = dataset_t[-2:].transpose()

    dataset_v = val_dataframe.transpose()
    feats_v = dataset_v[:-2].transpose()
    outs_v = dataset_v[-2:].transpose()
    return feats_t, outs_t, feats_v, outs_v


if __name__ == '__main__':
    config = {
        "data2_path": "data/sequoia_dataset_dsm.csv"
    }

    features_t, outcomes_t, features_v, outcomes_v = prepare_dataset(config["data2_path"])
    final_model = fit_survival_machine(features_t, outcomes_t)

    test_survival_machine(final_model, features_v, outcomes_v)