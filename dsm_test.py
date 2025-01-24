"""
  Created on Jan 2025
@author: Elena Markova
          for Attrition Rate Project
"""

import auton_survival
from auton_survival import estimators
from auton_survival.preprocessing import Preprocessor
from pandas import read_csv

from utils.custom_metric import calc_metric


def preprocess(_features):
    return Preprocessor().fit_transform(_features,
                                            cat_feats=['department', 'nationality', 'gender', 'family_status'],
                                            num_feats=[
                                                        "age",
                                                        "days_before_salary_increase",
                                                        "salary_increase",
                                                        "overtime",
                                                        "salary_6m_average",
                                                        "salary_cur"
                                                        ]
                                            )


def fit_survival_machine(_features, _outcomes):
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



def test_survival_machine(_model: auton_survival.models.dsm.DeepSurvivalMachines, _dataset):
    _dataset = _dataset.transpose()
    feats = _dataset[:-2].transpose()
    outs = _dataset[-2:].transpose()
    feats = preprocess(feats)
    times = []
    true_events = []
    predictions = []
    for t in outs.time:
        times.append(t)
    for e in outs.event:
        true_events.append(e)
    for n, f in enumerate(feats.transpose()):
        print(feats.iloc[[n]], times[n])
        pred = _model.predict_risk(feats.iloc[[n]], times=[times[n]])  # predict risk of event at ground-truth event time
        predictions.append(pred[0][0])

    calc_metric(predictions, true_events, 0.2)



def prepare_dataset(_data_path: str):
    dataset = read_csv(config["data_path"], delimiter=',')
    dataset = dataset.transpose()
    feats = dataset[:-2].transpose()
    outs = dataset[-2:].transpose()
    return feats, outs


if __name__ == '__main__':
    config = {
        "data_path": "data/dataset-nn-small-dsm.csv",  # dataset contains columns "event" and "time"
        "data2_path": "data/october-works-dsm.csv"
    }

    features, outcomes = prepare_dataset(config["data2_path"])
    final_model = fit_survival_machine(features, outcomes)

    test_dataset = read_csv(config["data_path"], delimiter=",")
    test_survival_machine(final_model, test_dataset)