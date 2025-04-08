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
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostClassifier, Pool

from pathlib import Path
# from examples.local import setting_dask_env

import seaborn as sn
import matplotlib.pyplot as plt

from utils.dataset import create_features_for_datasets, collect_datasets, minority_class_resample, prepare_dataset_2, get_united_dataset

# industry_avg_income = df.groupby('field')['income_shortterm'].mean().to_dict()
# df['industry_avg_income'] = df['field'].map(industry_avg_income)
# df['income_vs_industry'] = df['income_shortterm'] - df['industry_avg_income']
# position_median_income = df.groupby('department')['income_shortterm'].median().to_dict()
# df['position_median_income'] = df['department'].map(position_median_income)


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
        bootstrap_type='Bayesian',
        od_wait=5,
        depth=8,  # normally set it from 6 to 10
        eval_metric='Accuracy',
        random_seed=42,
        sampling_frequency='PerTreeLevel',
        random_strength=2,  # default: 1
        loss_function="Logloss",
        bagging_temperature=1
    )

    # perform parameters optimization by grid search method
    # grid = {'learning_rate': [0.03, 0.1, 0.01, 0.003], 'depth': [4,6,8,10], 'l2_leaf_reg': [1, 3, 5, 7, 9], 'od_wait': [5, 10, 20, 100], 'od_type': ['Iter', 'IncToDec']}
    # res = model.randomized_search(grid,
    #                         X=_x_train,
    #                         y=_y_train,
    #                         n_iter=500,
    #                         plot=True)
    # print(f"Best CatBoost params: {res}")

    model.fit(
        _x_train,
        _y_train,
        eval_set=(_x_test, _y_test),
        verbose=False,
        # sample_weight=_sample_weight,
        # plot=True,
        # cat_features=_cat_feats_encoded - do this if haven't encoded cat features
    )
    train_pool = Pool(data=_x_train, label=_y_train)
    shap_values = model.get_feature_importance(prettified=False, type='ShapValues', data=train_pool)
    feature_names = _x_train.columns
    #base_value = shap_values[0, -1]  # Последний столбец для всех samples одинаков
    #print(f"Base value (средняя вероятность класса 1): {base_value:.4f}")
    importance = ((shap_values[:, :-1])).mean(axis=0)
    importance_abs = (abs(shap_values[:, :-1])).mean(axis=0)
    for f, i in zip(feature_names, importance):
        print(f"{f}: {i}")
    sorted_idx = np.argsort(np.abs(importance))  # Indices from highest to lowest magnitude
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_shap = [importance[i] for i in sorted_idx]  # Signed values
    sorted_abs_shap = [importance_abs[i] for i in sorted_idx]  # Unsigned values
    colors = ['red' if val > 0 else 'blue' for val in sorted_shap]
    sorted_shap = [abs(s) for s in sorted_shap]  # Signed values
    plt.barh(sorted_features, sorted_shap, color=colors)
    plt.title('CatBoost Feature Importance')
    plt.show()
    plt.clf()
    plt.barh(sorted_features, sorted_abs_shap)
    plt.title('Catboost Feature Importance (unsigned absolute values)')
    plt.show()

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
    print(f"CatBoost result: F1 = {f1_united:.2f}, Recall = {recall_united:.2f}, Precision - {precision_united:.2f}")
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
        print(f"test on {len(trg)} samples: F1={f1:.2f}, Recall={r:.2f}, Precision={p:.2f}")

        predicted_classes = predictions  # (np.array(predictions) > 0.5).astype(int)
        result = confusion_matrix(trg, predicted_classes)
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
    #return np.array([w_0 if i == 1 else w_1 for i in _y_train.values])
    return np.array([1 if i == 1 else 1 for i in _y_train.values])


def main(_config: dict):
    data_path = config['dataset_src']
    datasets = collect_datasets(data_path)
    rand_states = range(5)  # [777, 42, 6, 1370, 5087]
    score = [0, 0, 0]

    if _config['calculated_features']:
        datasets, new_cat_feat = create_features_for_datasets(datasets)
        _config['cat_features'] += new_cat_feat

    for split_rand_state in rand_states:
        d_train, d_val, d_test, cat_feats_encoded = prepare_dataset_2(datasets, config['make_synthetic'], config['encode_categorical'], config['test_split'], config['cat_features'], split_rand_state)

        if _config['smote']:
            d_train = minority_class_resample(d_train, config['cat_features'])
            d_val = minority_class_resample(d_val, config['cat_features'])
            d_test = minority_class_resample(d_test, config['cat_features'])

        x_train, y_train, x_val, y_val = get_united_dataset(d_train, d_val, d_test)
        # x_train, x_test, y_train, y_test = prepare_dataset(dataset, config['test_split'], config['normalize'])

        print(f"X train: {x_train.shape[0]}, x_val: {x_val.shape[0]}, y_train: {y_train.shape[0]}, y_val: {y_val.shape[0]}")
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
    print(f"Final score of cross-val: F1={score[0]:.2f}, Recall = {score[1]:.2f}, Precision={score[2]:.2f}")

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
        'dataset_src': 'data/dsm',  # 2223_snap_randx2',
        'encode_categorical': True,
        'calculated_features': True,
        'make_synthetic': 'sdv',  # options: 'sdv', 'ydata', None
        'smote': True,  # perhaps not needed for catboost and in case if minority : majority > 0.5
        'cat_features': ['gender', 'citizenship', 'department', 'field']  #, 'occupational_hazards']
    }

    main(config)

# 97d0ae93-9dfc-4c2a-9183-a0420a4d0771

