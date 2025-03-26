"""
  Created on Jan 2025
@author: Elena Markova
          for Attrition Rate Project
"""

import os

os.environ['YDATA_LICENSE_KEY'] = '97d0ae93-9dfc-4c2a-9183-a0420a4d0771'

import pickle
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from pandas import read_csv
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostClassifier


from pathlib import Path
# from examples.local import setting_dask_env

from ydata.connectors import LocalConnector
from ydata.connectors.filetype import FileType
from ydata.metadata import Metadata
from ydata.synthesizers.regular.model import RegularSynthesizer
from ydata.utils.formats import read_json
from ydata.dataset.dataset import Dataset
from ydata.report import SyntheticDataProfile
from dask.dataframe.io import from_pandas, from_array

import seaborn as sn
import matplotlib.pyplot as plt

CATEGORICAL_FEATURES = ['gender', 'citizenship', 'department', 'field', 'occupational_hazards']


def train_xgboost_classifier(_x_train, _y_train, _x_test, _y_test, _sample_weight, _num_iters):
    model = xgb.XGBClassifier(objective="binary:logistic", random_state=42, early_stopping_rounds=15, eval_set=[(_x_test, _y_test)])
    best_f1 = 0.
    best_model = model
    test_result = {}

    print(f"Fitting XGBoost classifier...")
    for iter in range(_num_iters):
        model.fit(_x_train, _y_train, eval_set=[(_x_test, _y_test)], verbose=False)
        predictions = model.predict(_x_test)

        test_result['F1'] = f1_score(_y_test, predictions)
        if test_result['F1'] > best_f1:
            best_f1 = test_result['F1']
            best_model = model

            test_result['Recall'] = recall_score(_y_test, predictions)
            test_result['Precision'] = precision_score(_y_test, predictions)

    feature_importance = best_model.get_booster().get_score(importance_type='gain')
    keys = list(feature_importance.keys())

    # Debug print:
    values = list(feature_importance.values())
    for k, v in zip(keys, values):
        print(k)
    for k, v in zip(keys, values):
        print(v)

    data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by="score", ascending=False)
    l = data.nlargest(20, columns="score").plot(kind='barh', figsize=(20, 10))  # plot top 20 features
    print(l)
    print(f"XGBoost test result: Recall = {test_result['Recall']}\nPrecision = {test_result['Precision']}\nF1 = {test_result['F1']}")

    feature_importance = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': _x_train.columns, 'Importance': feature_importance})
    print(feature_importance_df)

    return best_model

def train_catboost(_x_train, _y_train, _x_test, _y_test, _sample_weight, _cat_feats_encoded, _num_iters):
    # model already initialized with latest version of optimized parameters for our dataset
    model = CatBoostClassifier(
        iterations=2000,  # default to 1000
        learning_rate=0.1,
        od_type='IncToDec',
        l2_leaf_reg=5,
        od_wait=5,
        depth=8,  # normally set it from 6 to 10
        eval_metric='F1',
        random_seed=42,
        sampling_frequency='PerTreeLevel',
        random_strength=2  # default: 1
    )

    # perform parameters optimization by grid search method
    # grid = {'learning_rate': [0.03, 0.1, 0.01, 0.003], 'depth': [4,6,8,10], 'l2_leaf_reg': [1, 3, 5, 7, 9]}  #, 'od_wait': [5, 10, 20, 100]}, 'od_type': ['Iter', 'IncToDec']}
    # res = model.randomized_search(grid,
    #                         X=_x_train,
    #                         y=_y_train,
    #                         n_iter=500,
    #                         plot=True)
    # print(f"Best CatBoost params: {res}")

    print(_cat_feats_encoded)
    model.fit(
        _x_train,
        _y_train,
        eval_set=(_x_test, _y_test),
        verbose=False,
        sample_weight=_sample_weight,
        # plot=True,
        # cat_features=_cat_feats_encoded - do this if haven't encoded cat features
    )
    print("Feature importance:", model.get_feature_importance(prettified=True))

    return model


def train_random_forest_regression(_x_train, _y_train, _x_test, _y_test, _sample_weight, _num_iters):
    model = RandomForestRegressor(n_estimators=100)
    best_precision = 0.
    best_model = model

    test_result = {}
    print(f"Fitting Random Forest...")
    for iter in range(_num_iters):
        model.fit(_x_train, _y_train, sample_weight=_sample_weight)
        predictions = model.predict(_x_test)

        test_result['r2_score'] = r2_score(_y_test, predictions)
        if test_result['r2_score'] > best_precision:
            best_precision = test_result['r2_score']
            best_model = model
            for i, y in enumerate(_y_test.values):
                print(_y_test.values[i], predictions[i])

    print(f"\nRandom Forest best result: R2 score = {test_result['r2_score']}\n")

    feature_importance = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': _x_train.columns, 'Importance': feature_importance})
    print(feature_importance_df)

    return best_model

def train_random_forest_regr(_x_train, _y_train, _x_test, _y_test, _sample_weight, _num_iters):
    model = RandomForestRegressor(n_estimators=100, max_features='sqrt')
    best_precision = 0.
    best_model = model

    print("Model attr:", model.__dict__)
    test_result = {}
    print(f"Fitting Random Forest...")
    for iter in range(_num_iters):
        model.fit(_x_train, _y_train, sample_weight=_sample_weight)
        print("\nModel attr after fitting:", model.__dict__)
        predictions = model.predict(_x_test)

        # Transform probabilities to binary classification output in order to calc metrics:
        thrs = 0.5
        for i, p in enumerate(predictions):
            if p > thrs:
                predictions[i] = 1
            else:
                predictions[i] = 0
        print(_y_test, predictions)
        test_result['Precision'] = precision_score(_y_test, predictions)
        if test_result['Precision'] > best_precision:
            best_precision = test_result['Precision']
            best_model = model

            test_result['R2'] = r2_score(_y_test, predictions)
            test_result['Recall'] = recall_score(_y_test, predictions)
            test_result['F1'] = f1_score(_y_test, predictions)

    print(f"\nRandom Forest best result: R2 score = {test_result['R2']}\nRecall = {test_result['Recall']}\nPrecision = {test_result['Precision']}\nF1 = {test_result['F1']}")

    feature_importance = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': _x_train.columns, 'Importance': feature_importance})
    print(feature_importance_df)

    return best_model


def train_random_forest_cls(_x_train, _y_train, _x_test, _y_test, _sample_weight, _num_iters):
    model = RandomForestClassifier(n_estimators=100,
                                   max_depth=3,
                                   max_features=1,
                                   min_samples_leaf=7,
                                   min_samples_split=7)
    best_f1 = 0.
    best_model = model

    test_result = {}
    # grid_space = {'max_depth': [3, 5, 10, None],
    #               'n_estimators': [50, 100, 200],
    #               'max_features': [1, 3, 5, 7, 9],
    #               'min_samples_leaf': [1, 2, 3, 7],
    #               'min_samples_split': [1, 2, 3, 7]
    #               }
    # grid = GridSearchCV(model, param_grid=grid_space, cv=3, scoring='f1')
    # model_grid = grid.fit(_x_train, _y_train)
    # print('Best hyperparameters are: ' + str(model_grid.best_params_))
    # print('Best score is: ' + str(model_grid.best_score_))

    print(f"Fitting Random Forest classifier...")
    for iter in range(_num_iters):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(_x_train, _y_train)  #, sample_weight=_sample_weight)
        predictions = model.predict(_x_test)

        test_result['Precision'] = precision_score(_y_test, predictions)
        test_result['Recall'] = recall_score(_y_test, predictions)
        test_result['F1'] = f1_score(_y_test, predictions)

        if test_result['F1'] > best_f1:
            best_f1 = test_result['F1']
            best_model = model

            test_result['Recall'] = recall_score(_y_test, predictions)
            test_result['Precision'] = precision_score(_y_test, predictions)

    print(f"\nRandom Forest classifier best result: Recall = {test_result['Recall']}\nPrecision = {test_result['Precision']}\nF1 = {test_result['F1']}")

    feature_importance = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': _x_train.columns, 'Importance': feature_importance})
    print(feature_importance_df)

    return best_model



def train(_x_train, _y_train, _x_test, _y_test, _sample_weight, _cat_feats_encoded, _model_name, _num_iters, _maximize):
    if _model_name == 'XGBoostClassifier':
        model = train_xgboost_classifier(_x_train, _y_train, _x_test, _y_test, _sample_weight, _num_iters)
    elif _model_name == 'RandomForestRegressor':
        model = train_random_forest_regr(_x_train, _y_train, _x_test, _y_test, _sample_weight, _num_iters)
    elif _model_name == 'RandomForestRegressor_2':
        model = train_random_forest_regression(_x_train, _y_train, _x_test, _y_test, _sample_weight, _num_iters)
    elif _model_name == "RandomForestClassifier":
        model = train_random_forest_cls(_x_train, _y_train, _x_test, _y_test, _sample_weight, _num_iters)
    elif _model_name == "CatBoostClassifier":
        model = train_catboost(_x_train, _y_train, _x_test, _y_test, _sample_weight, _cat_feats_encoded, _num_iters)
    else:
        print("Model name error: this model is not implemented yet!")
        return

    return model


def normalize(_data: pd.DataFrame):
    return _data.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)

def make_synthetic(_dataset: pd.DataFrame, _split_rand_state: int, _size: int = 1000):
    connector = LocalConnector()  # , keyfile_dict=token)

    # Instantiate a synthesizer
    cardio_synth = RegularSynthesizer()
    val = _dataset.sample(frac=0.3, random_state=_split_rand_state)
    test = val  # .sample(frac=0.3, random_state=SPLIT_RANDOM_STATE)
    trn = _dataset.drop(val.index)
    # memory_usage = trn.memory_usage(index=True, deep=False).sum()
    # npartitions = int(((memory_usage / 10e5) // 100) + 1)
    data = Dataset(trn)
    # calculating the metadata
    metadata = Metadata(data)

    # fit model to the provided data
    cardio_synth.fit(data, metadata)

    # Generate data samples by the end of the synth process
    synth_sample = cardio_synth.sample(_size)

    # TODO target variable validation
    profile = SyntheticDataProfile(
        data,
        synth_sample,
        metadata=metadata,
        target="status",
        data_types=cardio_synth.data_types)

    profile.generate_report(
        output_path="./cardio_report_example.pdf",
    )
    return synth_sample.to_pandas()  # {t: df.to_pandas() for t, df in res.items()}

def encode_categorical(_dataset: pd.DataFrame, _encoder: OneHotEncoder):
    encoded_features = _encoder.transform(_dataset[CATEGORICAL_FEATURES]).toarray().astype(int)
    encoded_df = pd.DataFrame(encoded_features, columns=_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    numerical_part = _dataset.drop(columns=CATEGORICAL_FEATURES)
    return pd.concat([encoded_df, numerical_part], axis=1), encoded_df

def collect_datasets(_data_path: str):
    datasets = []
    for filename in os.listdir(_data_path):
        if '.csv' not in filename:
            continue
        dataset_path = os.path.join(_data_path, filename)
        print(filename)
        dataset = pd.read_csv(dataset_path)
        print("cols before", dataset.columns)
        strings_to_drop = ['long',"birth","code",'external','overtime','termination','recruit']
        dataset = dataset.drop(
            columns=[c for c in dataset.columns if any(string in c for string in strings_to_drop)])
        print("cols after", dataset.columns)
        datasets.append(dataset)
    return datasets

def prepare_dataset_2(_datasets: list, _make_synthetic: bool, _encode_categorical: bool, _test_split: float, _split_rand_state: int):
    cat_feature_names = CATEGORICAL_FEATURES

    if _encode_categorical:
        concat_dataset = pd.concat(_datasets)
        encoder = OneHotEncoder()
        encoder.fit(concat_dataset[CATEGORICAL_FEATURES])

    if _make_synthetic:
        united = pd.concat(_datasets, axis=0)
        sample_df = make_synthetic(united, _split_rand_state, 1000)

        # Check for similar rows and print if match:
        for n, syn_row in sample_df.iterrows():
            for n1, real_row in united.iterrows():
                if (syn_row.values.astype(int) == real_row.values.astype(int)).all():
                    print("Match!", real_row.values.astype(int), syn_row.values.astype(int))

        if _encode_categorical:
            sample_df, _ = encode_categorical(sample_df, encoder)

    n = 0
    d_val = []
    d_train = []
    d_test = []
    for dataset in _datasets:
        n += 1
        # if _make_synthetic:
        #     sample_df = make_synthetic(dataset)

        if _encode_categorical:
            # Convert the encoded features to a DataFrame
            dataset, encoded_part = encode_categorical(dataset, encoder)

            cat_feature_names = encoded_part.columns.values

        #val = dataset.loc[val.index] # dataset.sample(frac=0.3, random_state=_split_rand_state)
        #test = val  # .sample(frac=0.3, random_state=_split_rand_state)
        #trn = dataset.drop(val.index)
        val = dataset.sample(frac=_test_split, random_state=_split_rand_state)
        test = val  # .sample(frac=0.3, random_state=_split_rand_state)
        trn = dataset.drop(val.index)

        # if _make_synthetic:
        #     trn = pd.concat([trn, sample_df], axis=0)
        # val = val.drop(test.index)

        d_val.append(val)
        d_train.append(trn)
        d_test.append(test)

    if _make_synthetic:
        d_train[0] = pd.concat([d_train[0], sample_df], axis=0)
    return d_train, d_val, d_test, cat_feature_names


def prepare_dataset(_dataset: pd.DataFrame, _test_split: float, _normalize: bool):
    target_idx = -1  # index of "works/left" column

    dataset = _dataset.transpose()
    trg = dataset[target_idx:]
    trn = dataset[:target_idx]

    # val_size = 2000
    # trn = trn.transpose()
    # trg = trg.transpose()
    # x_train = trn[val_size:]
    # x_test = trn[:val_size]
    # y_train = trg[val_size:]
    # y_test = trg[:val_size]

    x_train, x_test, y_train, y_test = train_test_split(trn.transpose(), trg.transpose(), test_size=_test_split)

    if _normalize:  # normalization is NOT needed for decision trees!
        x_train = normalize(x_train)
        x_test  = normalize(x_test)

    return x_train, x_test, y_train, y_test


def show_decision_tree(_model):
    tree_ = _model.estimators_[0].tree_
    feature_list = ["department_encoded",
                    "department_encoded2",
                    "department_encoded3",
                    "seniority_encoded",
                    "nationality_encoded",
                    "nationality_encoded2",
                    "nationality_encoded3",
                    "age_encoded",
                    "gender_encoded",
                    "gender_encoded2",
                    "gender_encoded3",
                    # "vacation_days_encoded",
                    "days_before_salary_increase_encoded",
                    "salary_increase_encoded",
                    "overtime_encoded",
                    "family_status_encoded",
                    "family_status_encoded2",
                    "family_status_encoded3",
                    "family_status_encoded4",
                    # "km_to_work_encoded",
                    "salary_6m_average_encoded",
                    "salary_cur_encoded"
                    ]
    feature_names = [feature_list[i] for i in tree_.feature]
    feature_name = [
        feature_names[i] for i in tree_.feature
    ]

    def recurse(node, depth):
        indent = "  " * depth
        if (tree_.feature[node] == -2):
            print("{}return {}".format(indent, tree_.value[node]))
        else:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)

    recurse(0, 1)


def test(_model, _test_data):
    test_data_all = pd.concat(_test_data, axis=0)
    t = test_data_all.transpose()
    trg = t[-1:].transpose()
    trn = t[:-1].transpose()
    threshold = 0.5

    print(f"Testing on united data with threshold = {threshold}...")
    predictions = _model.predict_proba(trn)
    predictions = (predictions[:, 1] > threshold).astype(int)

    f1_united = f1_score(trg, predictions)
    recall_united = recall_score(trg, predictions)
    precision_united = precision_score(trg, predictions)
    print(f"CatBoost result: F1 = {f1_united}, Recall = {recall_united}, Precision - {precision_united}")
    result = confusion_matrix(trg, predictions)
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(result, annot=True, annot_kws={"size": 16}, fmt='d')  # font size

    plt.show()
    plt.clf()

    print("Testing separately...")
    for t in _test_data:
        t = t.transpose()
        trg = t[-1:].transpose()
        trn = t[:-1].transpose()
        predictions = _model.predict(trn)
        f1 = f1_score(trg, predictions)
        r = recall_score(trg, predictions)
        p = precision_score(trg, predictions)
        print(f"test on {len(trg)} samples: F1={f1}, Recall={r}, Precision={p}")

        predicted_classes = predictions  # (np.array(predictions) > 0.5).astype(int)
        result = confusion_matrix(trg, predicted_classes)
        # recall = recall_score(trg, predicted_classes)
        # prec = precision_score(trg, predicted_classes)
        # print(recall, prec)
        sn.set(font_scale=1.4)  # for label size
        sn.heatmap(result, annot=True, annot_kws={"size": 16}, fmt='d')  # font size

        # plt.show()
    plt.clf()
    return f1_united, recall_united, precision_united

def calc_weights(_y_train: pd.DataFrame, _y_val: pd.DataFrame):
    w_0, w_1 = 0, 0
    for i in _y_train.values:
        w_0 += 1 - i.item()
        w_1 += i.item()

    tot = w_0 + w_1
    w_0 = w_0 / tot
    w_1 = w_1 / tot
    print(f"weights:", w_0, w_1)
    return np.array([w_0 if i == 1 else w_1 for i in _y_train.values])
    # return np.array([1 if i == 1 else 1 for i in _y_val.values])

def get_united_dataset(_d_train: list, _d_val: list, _d_test: list, _make_synthetic: bool):
    trn = pd.concat(_d_train, axis=0)
    vl = pd.concat(_d_val, axis=0)

    trn = trn.transpose()
    vl = vl.transpose()

    x_train = trn[:-1].transpose()
    x_val = vl[:-1].transpose()
    y_train = trn[-1:].transpose()
    y_val = vl[-1:].transpose()
    return x_train, y_train, x_val, y_val


def main(_config: dict):
    data_path = config['dataset_src']
    datasets = collect_datasets(data_path)
    rand_states = [42, 777, 6, 1370, 5087]
    score = [0, 0, 0]

    for split_rand_state in rand_states:
        d_train, d_val, d_test, cat_feats_encoded = prepare_dataset_2(datasets, config['make_synthetic'], config['encode_categorical'], config['test_split'], split_rand_state)

        x_train, y_train, x_val, y_val = get_united_dataset(d_train, d_val, d_test, config['make_synthetic'])
        # x_train, x_test, y_train, y_test = prepare_dataset(dataset, config['test_split'], config['normalize'])

        print(f"X train: {x_train.shape[0]}, x_val: {x_val.shape[0]}, y_train: {y_train.shape[0]}, y_val: {y_val.shape[0]}")
        print(y_train)
        sample_weight = calc_weights(y_train, y_val)

        # print(sample_weight)
        trained_model = train(x_train, y_train, x_val, y_val, sample_weight, cat_feats_encoded, config['model'], config['num_iters'], config['maximize'])
        f1, r, p = test(trained_model, d_test)
        score[0] += f1
        score[1] += r
        score[2] += p

        if config['model'] == 'RandomForestClassifier':
           show_decision_tree(trained_model)

    score = (score[0] / len(rand_states), score[1] / len(rand_states), score[2] / len(rand_states))
    print(f"Final score of cross-val: F1={score[0]}, Recall = {score[1]}, Precision={score[2]}")

    with open('model.pkl', 'wb') as f:
       print("Saving model..")
       pickle.dump(trained_model, f)


if __name__ == '__main__':
    # config_path = 'config.json'  # config file is used to store some parameters of the dataset
    config = {
        'model': 'CatBoostClassifier',  # options: 'RandomForestRegressor', 'RandomForestRegressor_2','RandomForestClassifier', 'XGBoostClassifier', 'CatBoostClassifier'
        'test_split': 0.25,  # validation/training split proportion
        'normalize': False,  # normalize input values or not
        'num_iters': 20,  # number of fitting attempts
        'maximize': 'Precision',  # metric to maximize
        'dataset_src': 'data/2223',
        'encode_categorical': True,
        'make_synthetic': False  # whether to use synthetic data
    }
    main(config)

# 97d0ae93-9dfc-4c2a-9183-a0420a4d0771