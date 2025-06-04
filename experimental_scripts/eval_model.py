"""
  Created on Jan 2025
@author: Elena Markova
          for Attrition Rate Project
"""

import pickle
from pandas import read_csv
from tensorflow import keras as K
from sklearn.metrics import r2_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder

import pandas as pd
import os
import seaborn as sn
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg', 'GTK3Agg', etc.
import matplotlib.pyplot as plt
from utils.dataset import collect_datasets
from utils.dataset import create_features_for_datasets, add_quality_features

CATEGORICAL_FEATURES = ['gender', 'citizenship', 'department', 'field', 'city', 'income_group', 'position_industry', 'region_population_group']

def encode_categorical(_dataset: pd.DataFrame, _encoder: OneHotEncoder):
    encoded_features = _encoder.transform(_dataset[CATEGORICAL_FEATURES]).toarray().astype(int)
    encoded_df = pd.DataFrame(encoded_features, columns=_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    numerical_part = _dataset.drop(columns=CATEGORICAL_FEATURES)
    return pd.concat([encoded_df, numerical_part], axis=1), encoded_df


def test_model(_model: K.Model, _feat: pd.DataFrame, _trg: pd.DataFrame):
    predictions = model.predict_proba(_feat)
    predictions = [int(p[1] > 0.7) for p in predictions]

    f1 = f1_score(_trg, predictions)
    r = recall_score(_trg, predictions)
    p = precision_score(_trg, predictions)

    print(f"F1 = {f1:.2f}, Recall = {r:.2f}, Precision = {p:.2f}")
    result = confusion_matrix(_trg, predictions)

    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(result, annot=True, annot_kws={"size": 16}, fmt='d')  # font size

    plt.show()


def test_rowwise(_model: K.Model, _dataset: pd.DataFrame):
    result_total = {}
    print(len(_dataset))
    for n, row in _dataset.iterrows():
        if row['status'] == 0:
            code = row['code']
            term_date = row['termination_date']
            sample = _dataset.loc[_dataset['code']==code]
            sample = sample.drop(columns=['recruitment_date', 'termination_date', 'code', 'birth_date'])
            sample = sample.drop(
                columns=[c for c in sample.columns if ('long' in c or 'external' in c or 'overtime' in c or 'index' in c)])
            sample = sample.transpose()

            feat = sample[:-1].transpose()
            trg = sample[-1:].transpose()

            pred = _model.predict(feat)
            result_total[code] = [term_date, pred[0]]

    for period in ['feb', 'may', 'aug', 'nov']:
        print("period", period)
        path = os.path.join('data', period)
        datasets = []
        for filename in os.listdir(path):
            print(f"Loading file {filename}...")
            datasets.append(read_csv(os.path.join(path, filename)))
        datasets = create_features_for_datasets(datasets)
        period_dataset = pd.concat(datasets, axis=0)
        print(len(period_dataset))

        encoder = OneHotEncoder()
        encoder.fit(period_dataset[CATEGORICAL_FEATURES])
        period_dataset = period_dataset.reset_index()
        period_dataset, cat_feature_names = encode_categorical(period_dataset, encoder)
        for n, row in period_dataset.iterrows():
            if row['status'] == 0:
                code = row['code']
                if code not in result_total.keys():
                    print("extra code")
                    continue
                term_date = row['termination_date']
                sample = period_dataset.loc[period_dataset['code'] == code]
                sample = sample.drop(columns=['recruitment_date', 'termination_date', 'external_factor', 'code', 'birth_date'])
                sample = sample.drop(
                    columns=[c for c in sample.columns if
                             ('long' in c or 'external' in c or 'overtime' in c or 'index' in c)])
                sample = sample.transpose()

                feat = sample[:-1].transpose()
                trg = sample[-1:].transpose()

                pred = _model.predict(feat)
                result_total[code] += [pred[0]]

    num_total_match = 0
    num_diff = 0
    num_single_val = 0
    for code in result_total.keys():
        if 0.0 in result_total[code] and 1.0 in result_total[code]:
            num_diff += 1
            print("Has difference", result_total[code])
        elif len(result_total[code]) == 2:
            print("Single", result_total[code])
            num_single_val += 1
        else:
            num_total_match += 1

    print(num_diff, num_total_match, num_single_val)



if __name__ == '__main__':
    config = {
        "test_data_path": "data/24_12",
        'train_data_path': 'data/2223_12',
        "model_path": "model.pkl"
    }

    model = pickle.load(open(config["model_path"], 'rb'))
    datasets = collect_datasets(config["test_data_path"])
    n_test_datasets = len(datasets)
    train_datasets = collect_datasets(config['train_data_path'])
    all_datasets = datasets + train_datasets

    all_datasets, _ = create_features_for_datasets(all_datasets)

    dataset_to_encode = pd.concat(train_datasets, axis=0)
    encoder = OneHotEncoder()
    encoder.fit(dataset_to_encode[CATEGORICAL_FEATURES])
    dataset_to_encode = dataset_to_encode.reset_index()

    test_datasets = all_datasets[:n_test_datasets]
    if n_test_datasets > 1:
        test_datasets.append(pd.concat(test_datasets, axis=0))
    # dataset, cat_feature_names = encode_categorical(dataset, encoder)
    # test_rowwise(model, dataset)

    print("Test on each dataset separately...")
    for d in test_datasets:
        strings_to_drop = ['long', 'birth', 'code', 'overtime', 'termination', 'recruit', 'index']
        d = d.drop(
            columns=[c for c in d.columns if any(string in c for string in strings_to_drop)])
        d = d.reset_index()
        print(d.columns)

        d, cat_feature_names = encode_categorical(d, encoder)
        d = d.transpose()

        feat = d[:-1].transpose()
        trg = d[-1:].transpose()

        # feat.insert(12, 'occupational_hazards_4', [0 for i in range(len(feat))])
        try:
            test_model(model, feat, trg)
        except Exception as e:
            print(e)
        d.transpose().to_excel("dataset_prepared_12.xlsx")