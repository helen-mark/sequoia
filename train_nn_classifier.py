import pandas as pd
import numpy as np
from pandas import read_csv
import tensorflow as tf
from tensorflow import keras as K

import eli5
from eli5.sklearn import PermutationImportance


def dataframe_to_dataset(dataframe: pd.DataFrame):
    dataframe = dataframe.copy()
    labels = dataframe.pop("status")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds


def encode_numerical_feature(feature: K.Input, name: str, dataset: tf.data.Dataset):
    # Create a Normalization layer for our feature
    normalizer = K.layers.Normalization()

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    normalizer.adapt(feature_ds)

    # Normalize the input feature
    encoded_feature = normalizer(feature)
    return encoded_feature


def encode_categorical_feature(feature: K.Input, name: str, dataset: tf.data.Dataset, is_string: bool):
    lookup_class = K.layers.StringLookup if is_string else K.layers.IntegerLookup
    lookup = lookup_class(output_mode="binary")

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the set of possible string values and assign them a fixed integer index
    lookup.adapt(feature_ds)

    # Turn the string input into integer indices
    encoded_feature = lookup(feature)
    return encoded_feature


def create_dataset(_dataset_path: str, _batch_size: int):
    batch_size = 32
    # load the dataset
    dataset = read_csv(_dataset_path, delimiter=',')
    dataset.head()
    val_dataframe = dataset.sample(frac=0.2, random_state=1337)
    train_dataframe = dataset.drop(val_dataframe.index)

    print(
        f"Using {len(train_dataframe)} samples for training "
        f"and {len(val_dataframe)} for validation"
    )

    train_ds = dataframe_to_dataset(train_dataframe)
    val_ds = dataframe_to_dataset(val_dataframe)
    val_ds_2 = dataframe_to_dataset(val_dataframe)

    train_ds = train_ds.batch(_batch_size)
    val_ds = val_ds.batch(_batch_size)
    return train_ds, val_ds, val_ds_2, train_dataframe, val_dataframe


def prepare_all_features(_train_ds: tf.data.Dataset):

    # Categorical features encoded as integers
    gender = K.Input(shape=(1,), name="gender", dtype="int64")
    department = K.Input(shape=(1,), name="department", dtype="int64")
    nationality = K.Input(shape=(1,), name="nationality", dtype="int64")
    family_status = K.Input(shape=(1,), name="family_status", dtype="int64")

    # Numerical features
    age = K.Input(shape=(1,), name="age")
    seniority = K.Input(shape=(1,), name="seniority")
    # vacation_days = K.Input(shape=(1,), name="vacation_days")
    days_before_salary_increase = K.Input(shape=(1,), name="days_before_salary_increase")
    salary_increase = K.Input(shape=(1,), name="salary_increase")
    overtime = K.Input(shape=(1,), name="overtime")
    # km_to_work = K.Input(shape=(1,), name="km_to_work")
    salary_6m_average = K.Input(shape=(1,), name="salary_6m_average")
    salary_cur = K.Input(shape=(1,), name="salary_cur")

    all_inputs = [
        department,
        seniority,
        nationality,
        age,
        gender,
        # vacation_days,
        days_before_salary_increase,
        salary_increase,
        overtime,
        family_status,
        # km_to_work,
        salary_6m_average,
        salary_cur
    ]

    # Integer categorical features
    gender_encoded = encode_categorical_feature(gender, "gender", _train_ds, False)
    department_encoded = encode_categorical_feature(department, "department", _train_ds, False)
    nationality_encoded = encode_categorical_feature(nationality, "nationality", _train_ds, False)
    family_status_encoded = encode_categorical_feature(family_status, "family_status", _train_ds, False)

    # Numerical features
    age_encoded = encode_numerical_feature(age, "age", _train_ds)
    seniority_encoded = encode_numerical_feature(seniority, "seniority", _train_ds)
    # vacation_days_encoded = encode_numerical_feature(vacation_days, "vacation_days", train_ds)
    days_before_salary_increase_encoded = encode_numerical_feature(days_before_salary_increase,
                                                                   "days_before_salary_increase", _train_ds)
    salary_increase_encoded = encode_numerical_feature(salary_increase, "salary_increase", _train_ds)
    overtime_encoded = encode_numerical_feature(overtime, "overtime", _train_ds)
    # km_to_work_encoded = encode_numerical_feature(km_to_work, "km_to_work", train_ds)
    salary_6m_average_encoded = encode_numerical_feature(salary_6m_average, "salary_6m_average", _train_ds)
    salary_cur_encoded = encode_numerical_feature(salary_cur, "salary_cur", _train_ds)

    all_features = K.layers.concatenate(
        [
            department_encoded,
            seniority_encoded,
            nationality_encoded,
            age_encoded,
            gender_encoded,
            # vacation_days_encoded,
            days_before_salary_increase_encoded,
            salary_increase_encoded,
            overtime_encoded,
            family_status_encoded,
            # km_to_work_encoded,
            salary_6m_average_encoded,
            salary_cur_encoded
        ]
    )
    return all_features, all_inputs


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


def create_model(_all_features, _all_inputs):
    n_neurons = 32
    x = K.layers.Dense(n_neurons, activation="relu")(_all_features)
    x = K.layers.Dropout(0.5)(x)
    output = K.layers.Dense(1, activation="sigmoid")(x)
    return K.Model(_all_inputs, output)


def train(_dataset_path: str, _batch_size: int):
    train_ds, val_ds, val_ds_2, train_dataframe, _  = create_dataset(_dataset_path, _batch_size)
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
        '/home/elena/ATTRITION/model.tf',
        monitor="val_loss",
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
        save_freq="epoch",
        initial_value_threshold=None,
    )
    # fit the keras model
    model.fit(train_ds, epochs=120, batch_size=_batch_size, validation_data=val_ds, callbacks=[tboard, save_best])

    print(model.summary())
    print('Evaluation...')
    model.evaluate(val_ds)

    calculate_feature_importance(model, train_dataframe, val_ds, val_ds_2)


if '__main__' == __name__:
    dataset_path = 'data/dataset-nn-small.csv'
    batch_size = 32
    train(dataset_path, batch_size)
