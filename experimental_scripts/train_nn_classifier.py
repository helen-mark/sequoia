"""
  Created on Jan 2025
@author: Elena Markova
          for Attrition Rate Project
"""
import sys
from os import path
from pathlib import Path

import eli5
import numpy as np
import pandas as pd
import tensorflow as tf
from pandas import read_csv
from tensorflow import keras as K
from sklearn.metrics import confusion_matrix, recall_score, precision_score
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

from kerastuner import HyperModel
from kerastuner.tuners import BayesianOptimization

sequoia = Path(__file__).parent.resolve()
sys.path.append(sequoia)

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


def permutation_feature_importance(_model: K.Model, _val_ds_batched: tf.data.Dataset, _val_ds: tf.data.Dataset, _feature_names: []):
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
    for i, f in enumerate(feature_importances):
        print(f"{_feature_names[i].name}: {f:.4f}")

    # perm = PermutationImportance(model, scoring=score, random_state=1).fit(X, Y)
    # eli5.show_weights(perm, feature_names=train_ds.columns.tolist())


def calculate_feature_importance(_model: K.Model, _train_dataframe: pd.DataFrame, _val_ds_batched: tf.data.Dataset, _val_ds: tf.data.Dataset, _features: []):
    feature_importance_by_model_coef(_model, _train_dataframe)
    permutation_feature_importance(_model, _val_ds_batched, _val_ds, _features)


def build_model(_hp):  # function for automatic hyperparams tuning
    input_layers = {}
    for feature in ['gender', 'department', 'citizenship', 'income_shortterm', 'income_longterm',
                    'absenteeism_shortterm', 'absenteeism_longterm', 'vacation_days_shortterm',
                    'vacation_days_longterm', 'overtime_shortterm', 'overtime_longterm', 'age', 'seniority',
                    'external_factor_1', 'external_factor_2', 'external_factor_3']:
        input_layers[feature] = tf.keras.layers.Input(name=feature, shape=(1,))

    # Объединяем входные слои
    concat_layer = tf.keras.layers.Concatenate()(list(input_layers.values()))

    # Создаём модель с объединённым входным слоем
    x = tf.keras.layers.Dense(units=_hp.Int('units', min_value=32, max_value=512, step=32), activation='relu')(
        concat_layer)
    x = tf.keras.layers.Dropout(rate=_hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1))(x)
    x = tf.keras.layers.Dense(units=_hp.Int('units', min_value=32, max_value=512, step=32), activation='relu')(x)
    x = tf.keras.layers.Dense(units=1, activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=list(input_layers.values()), outputs=x)

    optimizer = tf.keras.optimizers.Adam(learning_rate=_hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3, 1e-4]))
    model.compile(loss='binary_focal_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

def create_model(_all_features_reg: tf.Tensor, _all_features_nonreg: tf.Tensor, _all_inputs_reg: list, _all_inputs_nonreg: list):
    n_neurons_reg = 16
    n_neurons_nonreg = 32
    n_neurons_hidden = 16

    # Слои для каждой группы
    # regularized_layer = K.layers.Dense(n_neurons_reg, kernel_regularizer=K.regularizers.l2(0.01))(_all_features_reg)
    non_regularized_layer = K.layers.Dense(n_neurons_nonreg)(_all_features_nonreg)

    # Объединение выходов
    # concat_layer = K.layers.concatenate([regularized_layer, non_regularized_layer])
#    all_inputs = K.layers.concatenate(_all_inputs_reg + _all_inputs_nonreg)

    x = K.layers.Dense(n_neurons_nonreg, activation='relu')(_all_features_nonreg)
    x = K.layers.Dropout(0.5)(x)
   # x = K.layers.Dense(n_neurons, activation="relu")(concat_layer)
    x = K.layers.Dense(n_neurons_hidden, activation="relu")(x)
    x = K.layers.Dense(n_neurons_hidden/2, activation="relu")(x)
    x = K.layers.Dropout(0.25)(x)
    output = K.layers.Dense(1, activation="sigmoid")(x)

    model = K.Model(_all_inputs_nonreg, output)

    optimizer = K.optimizers.Adam(learning_rate=0.01)
    model.compile(loss='binary_focal_crossentropy', optimizer=optimizer, metrics=['accuracy', K.metrics.Recall(), K.metrics.Precision()])
    return model


def train(_dataset1: pd.DataFrame, _val_dataset: pd.DataFrame, _batch_size: int):
    train_ds, val_ds, val_ds_2, train_dataframe, _ = create_dataset(_dataset1, _val_dataset, _batch_size)
    all_features_reg, all_features_nonreg, all_inputs_reg, all_inputs_nonreg = prepare_all_features(train_ds)
    model = create_model(all_features_reg, all_features_nonreg, all_inputs_reg, all_inputs_nonreg)

    # tuner = BayesianOptimization(build_model,
    #                              objective='val_accuracy',
    #                              max_trials=10,
    #                              directory='my_dir',
    #                              project_name='bayesian_tune')
    #
    # # Обучение модели
    # tuner.search(train_ds,epochs=100,  validation_data=val_ds)
    #
    # # Получение результатов
    # best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    # best_model = tuner.hypermodel.build(best_hps)
    # best_model.fit(train_ds,epochs=100,  validation_data=val_ds)
    # print("Best hyperparam:", best_hps)

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

    def scheduler(epoch, lr):
        if epoch < 30:
            return lr
        else:
            return lr * np.exp(-0.02)

    scheduler = K.callbacks.LearningRateScheduler(scheduler)
    # fit the keras model
    model.fit(train_ds, epochs=100, batch_size=_batch_size, validation_data=val_ds, callbacks=[tboard, save_best, scheduler])

    best_model = K.models.load_model('best_keras_model')
    print(best_model.summary())
    # print('Evaluation...')
    # best_model.evaluate(val_ds)

    calculate_feature_importance(best_model, train_dataframe, val_ds, val_ds_2, all_features_nonreg)
    return best_model


def evaluation_and_matrix(_val_1: pd.DataFrame, _val_2: pd.DataFrame, _val_common: pd.DataFrame, _batch_size: int):
    model = K.models.load_model('best_keras_model')
    print(model.summary())
    # evaluate on different production fields separately
    val_ds_1 = dataframe_to_dataset(_val_1)
    val_ds_1 = val_ds_1.batch(_batch_size)
    val_ds_2 = dataframe_to_dataset(_val_2)
    val_ds_2 = val_ds_2.batch(_batch_size)

    print("evaluate on polimer...")
    model.evaluate(val_ds_1)
    model.evaluate(val_ds_2)
    # Predict
    val_ds = dataframe_to_dataset(_val_common)
    val_ds_b = val_ds.batch(_batch_size)

    PREDICTIONS = []
    TRUE_LABELS = []
    for n, row in _val_common.iterrows():
        df = _val_common.loc[_val_common.index == n]
        df = df.loc[df['age'] == row['age']]
        true_label = row['status']
        ds = dataframe_to_dataset(df)
        ds = ds.batch(_batch_size)
        pred = model.predict(ds, verbose=False)
        PREDICTIONS.append(pred[0][0])
        TRUE_LABELS.append(true_label)

    predicted_classes = (np.array(PREDICTIONS) > 0.5).astype(int)
    result = confusion_matrix(TRUE_LABELS, predicted_classes)
    recall = recall_score(TRUE_LABELS, predicted_classes)
    prec = precision_score(TRUE_LABELS, predicted_classes)
    print(recall, prec)
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(result, annot=True, annot_kws={"size": 16})  # font size

    plt.show()


if '__main__' == __name__:
    dataset_path = 'data/sequoia_dataset.csv'
    dataset2_path = 'data/polimer_dataset.csv'

    dataset1 = pd.read_csv(dataset_path)
    dataset2 = pd.read_csv(dataset2_path)

    val_1 = dataset1.sample(frac=0.2, random_state=1337)
    val_2 = dataset2.sample(frac=0.2, random_state=1337)
    train_1 = dataset1.drop(val_1.index)
    train_2 = dataset2.drop(val_2.index)

    # dataset = pd.concat([dataset1, dataset2], axis=0)

    val_dataframe = pd.concat([val_1, val_2], axis=0)  # dataset.sample(frac=0.2, random_state=1337)
    train_dataframe = pd.concat([train_1, train_2], axis=0)  # dataset.drop(val_dataframe.index)

    batch_size = 32
    best_model = train(train_dataframe, val_dataframe, batch_size)

    evaluation_and_matrix(val_1, val_2, val_dataframe,  batch_size)

