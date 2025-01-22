"""
  Created on Jan 2025
@author: Elena Markova
          for Attrition Rate Project
"""

import pickle
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from pandas import read_csv
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn import tree


def train_xgboost_classisier(_x_train, _y_train, _x_test, _y_test, _num_iters):
    model = xgb.XGBClassifier(objective="binary:logistic", random_state=42, early_stopping_rounds=15, eval_set=[(_x_test, _y_test)])
    best_precision = 0.
    best_model = model
    test_result = {}

    print(f"Fitting XGBoost classifier...")
    for iter in range(_num_iters):
        model.fit(_x_train, _y_train, eval_set=[(_x_test, _y_test)], verbose=False)
        predictions = model.predict(_x_test)

        test_result['Precision'] = precision_score(_y_test, predictions)
        if test_result['Precision'] > best_precision:
            best_precision = test_result['Precision']
            best_model = model

            test_result['R2'] = r2_score(_y_test, predictions)
            test_result['Recall'] = recall_score(_y_test, predictions)
            test_result['F1'] = f1_score(_y_test, predictions)

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
    print(f"XGBoost test result: R2 score = {test_result['R2']}\nRecall = {test_result['Recall']}\nPrecision = {test_result['Precision']}\nF1 = {test_result['F1']}")

    return best_model


def train_random_forest_regr(_x_train, _y_train, _x_test, _y_test, _num_iters):
    model = RandomForestRegressor(n_estimators=10, max_features='sqrt')
    thrs = 0.5
    best_precision = 0.
    best_model = model

    print("Model attr:", model.__dict__)
    test_result = {}
    print(f"Fitting Random Forest...")
    for iter in range(_num_iters):
        model.fit(_x_train, _y_train)
        print("\nModel attr after fitting:", model.__dict__)
        predictions = model.predict(_x_test)

        # Transform probabilities to binary classification output in order to calc metrics:
        for i, p in enumerate(predictions):
            if p > thrs:
                predictions[i] = 1
            else:
                predictions[i] = 0

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


def train_random_forest_cls(_x_train, _y_train, _x_test, _y_test, _num_iters):
    model = RandomForestClassifier(n_estimators=10, max_features='sqrt')
    thrs = 0.5
    best_precision = 0.
    best_model = model

    print("Model attr:", model.__dict__)
    test_result = {}

    print(f"Fitting Random Forest classifier...")
    for iter in range(_num_iters):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(_x_train, _y_train)
        print("\nModel attr after fitting:", model.__dict__)
        print("\nModel get_param:", model.get_params(deep=True))
        predictions = model.predict(_x_test)

        test_result['Precision'] = precision_score(_y_test, predictions)
        if test_result['Precision'] > best_precision:
            best_precision = test_result['Precision']
            best_model = model

            test_result['Recall'] = recall_score(_y_test, predictions)
            test_result['F1'] = f1_score(_y_test, predictions)

    print(f"\nRandom Forest classifier best result: Recall = {test_result['Recall']}\nPrecision = {test_result['Precision']}\nF1 = {test_result['F1']}")

    feature_importance = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': _x_train.columns, 'Importance': feature_importance})
    print(feature_importance_df)

    return best_model


def train_linear_regression(_x_train, _y_train, _x_test, _y_test, _num_iters):
    model = LinearRegression()
    thrs = 0.5
    best_precision = 0.
    best_model = model
    test_result = {}

    print(f"Fitting Linear Regression...")
    for iter in range(_num_iters):
        model.fit(_x_train, _y_train)
        predictions = model.predict(_x_test)

        # Transform probabilities to binary classification output in order to calc metrics:
        for i, p in enumerate(predictions):
            if p > thrs:
                predictions[i] = 1
            else:
                predictions[i] = 0

        test_result['Precision'] = precision_score(_y_test, predictions)
        if test_result['Precision'] > best_precision:
            best_precision = test_result['Precision']
            best_model = model

            test_result['R2'] = r2_score(_y_test, predictions)
            test_result['Recall'] = recall_score(_y_test, predictions)
            test_result['F1'] = f1_score(_y_test, predictions)

    print(f"Linear Regression best result: R2 score = {test_result['R2']}\nRecall = {test_result['Recall']}\nPrecision = {test_result['Precision']}\nF1 = {test_result['F1']}")

    coefficients = pd.concat([pd.DataFrame(_x_train.columns), pd.DataFrame(np.transpose(best_model.coef_))], axis=1)
    print(coefficients)
    return best_model


def train(_x_train, _y_train, _x_test, _y_test, _model_name, _num_iters, _maximize):
    if _model_name == 'XGBoostClassifier':
        model = train_xgboost_classisier(_x_train, _y_train, _x_test, _y_test, _num_iters)
    elif _model_name == 'RandomForestRegressor':
        model = train_random_forest_regr(_x_train, _y_train, _x_test, _y_test, _num_iters)
    elif _model_name == 'LinearRegression':
        model = train_linear_regression(_x_train, _y_train, _x_test, _y_test, _num_iters)
    elif _model_name == "RandomForestClassifier":
        model = train_random_forest_cls(_x_train, _y_train, _x_test, _y_test, _num_iters)
    else:
        print("Model name error: this model is not implemented yet!")
        return

    return model
    # models = [
    #           # KNeighborsRegressor(n_neighbors=3),
    #           # SVR(kernel='linear'),
    #           # LogisticRegression(),
    #           ]

def normalize(_data: pd.DataFrame):
    return _data.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)

def prepare_dataset(_dataset_path: str, _test_split: float, _normalize: bool):
    dataset = read_csv(_dataset_path, delimiter=',')
    dataset.head()
    target_idx = -1  # index of "works/left" column

    dataset = dataset.transpose()
    trg = dataset[target_idx:]
    trn = dataset[:target_idx]

    x_train, x_test, y_train, y_test = train_test_split(trn.transpose(), trg.transpose(), test_size=_test_split)
    print("y_test", y_test)

    if _normalize:
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


if __name__ == '__main__':
    # config_path = 'config.json'  # config file is used to store some parameters of the dataset
    config = {
        'model': 'RandomForestClassifier',  # options: 'RandomForestRegressor', 'RandomForestClassifier', 'XGBoostClassifier', 'LinearRegression'
        'test_split': 0.2,  # validation/training split proportion
        'normalize': False,  # normalize input values or not
        'num_iters': 20,  # number of fitting attempts
        'maximize': 'Precision'  # metric to maximize
    }
    dataset_src = 'data/dataset-2-small.csv'

    # Load the dataset from file and split it to train/test:
    x_train, x_test, y_train, y_test = prepare_dataset(dataset_src, config['test_split'], config['normalize'])

    # Train model and save it:
    model = train(x_train, y_train, x_test, y_test, config['model'], config['num_iters'], config['maximize'])
    show_decision_tree(model)

    with open('model.pkl', 'wb') as f:
       print("Saving model..")
       pickle.dump(model, f)
