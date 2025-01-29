"""
  Created on Jan 2025
@author: Elena Markova
          for Attrition Rate Project
"""

import eli5
import numpy as np
import pandas as pd
import tensorflow as tf
from pandas import read_csv
from tensorflow import keras as K

from utils.dataset import create_dataset, dataframe_to_dataset, encode_numerical_feature, encode_categorical_feature, prepare_all_features


def feature_importance_by_model_coef(_model: K.Model, _train_dataframe: pd.DataFrame):
    layer = _model.layers[-3]  # take layer with direct connections to input features
    feature_df = pd.DataFrame(columns=['feature', 'layer', 'neuron', 'weight', 'abs_weight'])

    w = layer.get_weights()
    w = np.array(w[0])
    n = 0
    _train_dataframe.pop('status')
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
    f_tot = {}
    abs_f_tot = {}
    for neuron in w.T:
        for f, name in zip(neuron, feature_list):
            feature_df.loc[len(feature_df)] = [name, 27, n, f, abs(f)]
            if name in f_tot.keys():
                f_tot[name] += f
                abs_f_tot[name] += abs(f)
            else:
                f_tot[name] = f
                abs_f_tot[name] = abs(f)

        n += 1

    for name in f_tot.keys():
        print(name, f_tot[name], abs_f_tot[name])

    feature_df = feature_df.sort_values(by=['abs_weight'])
    feature_df.reset_index(inplace=True)
    feature_df = feature_df.drop(['index'], axis=1)
    # print(feature_df)


def permutation_feature_importance(_model: K.Model, _val_ds_batched: tf.data.Dataset, _val_ds: tf.data.Dataset):
    def score(x, y):
        y_pred = _model.predict(_val_ds_batched)
        for i, v in enumerate(y_pred):
            if v > 0.5:
                y_pred[i] = 1
            else:
                y_pred[i] = 0
        m = K.metrics.Accuracy()
        m.update_state(np.float32(y), y_pred)
        return m.result()

    # fig = px.bar(feature_df, x='feature', y='abs_weight', template='simple_white')
    # fig.show()
    X = []
    Y = []
    for x, y in _val_ds:
        X.append([*x.values()])
        Y.append([y])

    # X = np.concatenate(X, axis=0)
    # Y = np.concatenate(Y, axis=0)

    base_score, score_decreases = eli5.permutation_importance.get_score_importances(score, np.array(X), np.array(Y))
    feature_importances = np.mean(score_decreases, axis=0)
    print(feature_importances)

    # perm = PermutationImportance(model, scoring=score, random_state=1).fit(X, Y)
    # eli5.show_weights(perm, feature_names=train_ds.columns.tolist())


def calculate_feature_importance(_model: K.Model, _train_dataframe: pd.DataFrame, _val_ds_batched: tf.data.Dataset, _val_ds: tf.data.Dataset):
    feature_importance_by_model_coef(_model, _train_dataframe)
    permutation_feature_importance(_model, _val_ds_batched, _val_ds)


def create_model(_all_features: tf.Tensor, _all_inputs: list):
    n_neurons = 32
    x = K.layers.Dense(n_neurons, activation="relu")(_all_features)
    x = K.layers.Dense(n_neurons/2, activation="relu")(x)
    x = K.layers.Dropout(0.5)(x)
    output = K.layers.Dense(1, activation="sigmoid")(x)
    return K.Model(_all_inputs, output)


def train(_dataset_path: str, _batch_size: int):
    train_ds, val_ds, val_ds_2, train_dataframe, _ = create_dataset(_dataset_path, _batch_size)
    all_features, all_inputs = prepare_all_features(train_ds)
    model = create_model(all_features, all_inputs)

    optimizer = K.optimizers.Adam(learning_rate=0.01)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', K.metrics.Recall(), K.metrics.Precision()])

    tboard = K.callbacks.TensorBoard(
        log_dir="logs",
        histogram_freq=0,
        write_graph=True,
        write_images=False,
        write_steps_per_second=False,
        update_freq="epoch",
        profile_batch=0,
        embeddings_freq=0,
        embeddings_metadata=None,
    )
    save_best = K.callbacks.ModelCheckpoint(
        'best_keras_model',
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
        save_freq="epoch",
        initial_value_threshold=None,
    )
    # fit the keras model
    model.fit(train_ds, epochs=100, batch_size=_batch_size, validation_data=val_ds, callbacks=[tboard, save_best])

    print(model.summary())
    print('Evaluation...')
    model.evaluate(val_ds)

    calculate_feature_importance(model, train_dataframe, val_ds, val_ds_2)


if '__main__' == __name__:
    dataset_path = 'data/october_works.csv'
    batch_size = 32
    train(dataset_path, batch_size)
