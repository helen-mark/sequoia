"""
  Created on Jan 2025
@author: Elena Markova
          for Attrition Rate Project
"""
import sys
from pathlib import Path

# Add project/ to Python path
sys.path.append(str(Path(__file__).parent.parent))

import auton_survival

import pandas as pd
from auton_survival import estimators
from auton_survival.preprocessing import Preprocessor
from pandas import read_csv

from utils.custom_metric import calc_metric
from utils.dataset import create_features_for_datasets, collect_datasets, minority_class_resample, prepare_dataset_2, get_united_dataset


TIME = 'time'
EVENT = 'event'

def preprocess(_features: pd.DataFrame, _cat_features: []):
    all_features = _features.columns.tolist()
    num_features = [f for f in all_features if f not in _cat_features]
    print(_features)
    return Preprocessor().fit_transform(_features,
                                            cat_feats=_cat_features,
                                            num_feats=num_features
                                            )

_VALID_MODELS = ['rsf', 'cph', 'dsm', 'dcph', 'dcm']
def fit_survival_machine(_features: pd.DataFrame, _outcomes: pd.DataFrame):
    model = estimators.SurvivalModel(model='dsm', distribution='LogNormal', temperature=1.0, layers=[128, 64], batch_size=64, learning_rate=1e-1, epochs=50)

    #model = estimators.SurvivalModel(model='cph', layers=[100, 50, 50], batch_size=64, learning_rate=1e-6, epochs=150)
    # model = estimators.SurvivalModel(model='rsf',
    #                                 n_estimators=100,
    #                                 max_depth=5,
    #                                 min_samples_split=7,
    #                                 min_samples_leaf=7,
    #                                 min_weight_fraction_leaf=1,
    #                                 max_features="sqrt",
    #                                 max_leaf_nodes=3,
    #                                 bootstrap=True,
    #                                 oob_score=False,
    #                                 n_jobs=None,
    #                                 random_state=None,
    #                                 verbose=0,
    #                                 warm_start=True,
    #                                 max_samples=7,
    #                                 low_memory=False,)  # Random survival forest

    # Preprocessor does both imputing and scaling of data:
    _features = _features.replace({True: 1, False: 0})  # replace encoded categorical features values
    print(_features)
    print("Fitting...")
    model.fit(_features, _outcomes)

    # metrics = ['brs', 'ibs', 'auc', 'ctd']
    # score = survival_regression_metric(metric='brs', _outcomes,
    #                                    _outcomes, predictions,
    #                                    times=[20])

    return model



def test_survival_machine(_model: auton_survival.models.dsm.DeepSurvivalMachines, _feats, _outs, _cat_feat):
    times = []
    true_events = []
    predictions = []
    _feats = _feats.replace({True: 1, False: 0})  # replace encoded categorical features values

    for t in _outs.time:
        times.append(t)
    for e in _outs.event:
        true_events.append(e)
    for n, row in _feats.iterrows():
        pred = _model.predict_risk(_feats.iloc[[n]], times=[times[n]])  # predict risk of event at ground-truth event time
        predictions.append(pred[0][0])
    calc_metric(predictions, true_events, 0.5)


def get_united_dsm_dataset(_d_train: [], _d_val: [], _d_test: []):
    trn = pd.concat(_d_train, axis=0)
    val = pd.concat(_d_val, axis=0)
    tst = pd.concat(_d_test, axis=0)

    trn = trn.rename(columns={'seniority': 'time', 'status': 'event'})
    val = val.rename(columns={'seniority': 'time', 'status': 'event'})
    tst = tst.rename(columns={'seniority': 'time', 'status': 'event'})

    out_trn = trn[[EVENT, TIME]]
    out_val = val[[EVENT, TIME]]
    out_tst = tst[[EVENT, TIME]]

    out_trn[TIME] = (out_trn[TIME] * 12)#.astype(int)
    out_val[TIME] = (out_val[TIME] * 12)#.astype(int)
    out_tst[TIME] = (out_tst[TIME] * 12)#.astype(int)

    print(out_trn)
    print(out_val)

    feat_trn = trn.drop(columns=[EVENT, TIME])
    feat_val = val.drop(columns=[EVENT, TIME])
    feat_tst = tst.drop(columns=[EVENT, TIME])

    return feat_trn, out_trn, feat_val, out_val, feat_tst, out_tst


def main(_config: dict):
    data_path = config['dataset_src']
    datasets = collect_datasets(data_path)
    rand_states = range(1)  # [777, 42, 6, 1370, 5087]
    score = [0, 0, 0]

    if _config['calculated_features']:
        datasets, new_cat_feat = create_features_for_datasets(datasets)
        _config['cat_features'] += new_cat_feat

    for split_rand_state in rand_states:
        d_train, d_val, d_test, cat_feats_encoded = prepare_dataset_2(datasets, config['make_synthetic'], False, config['test_split'], config['cat_features'], split_rand_state)

        if _config['smote']:
            d_train = minority_class_resample(d_train, config['cat_features'])
            d_val = minority_class_resample(d_val, config['cat_features'])
            d_test = minority_class_resample(d_test, config['cat_features'])

        x_train, y_train, x_val, y_val, _, _ = get_united_dsm_dataset(d_train, d_val, d_test)
        # x_train, x_test, y_train, y_test = prepare_dataset(dataset, config['test_split'], config['normalize'])

        print(f"X train: {x_train.shape[0]}, x_val: {x_val.shape[0]}, y_train: {y_train.shape[0]}, y_val: {y_val.shape[0]}")

        x_train = preprocess(x_train, cat_feats_encoded)
        x_val = preprocess(x_val, cat_feats_encoded)

        trained_model = fit_survival_machine(x_train, y_train)
        test_survival_machine(trained_model, x_val, y_val, cat_feats_encoded)



if __name__ == '__main__':
    config = {
        'model': 'CatBoostClassifier',  # options: 'RandomForestRegressor', 'RandomForestRegressor_2','RandomForestClassifier', 'XGBoostClassifier', 'CatBoostClassifier'
        'test_split': 0.25,  # validation/training split proportion
        'num_iters': 20,  # number of fitting attempts
        'maximize': 'Precision',  # metric to maximize
        'dataset_src': 'data/dsm',
        'calculated_features': True,
        'make_synthetic': None,  # options: 'sdv', 'ydata', None
        'smote': False,  # perhaps not needed for catboost and in case if minority : majority > 0.5
        'cat_features': ['gender', 'citizenship', 'department', 'field', 'city']  #, 'occupational_hazards']
    }
    main(config)
