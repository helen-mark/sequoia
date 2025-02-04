import tensorflow as tf
import tensorflow.keras as K
import pandas as pd
from pandas import read_csv


def dataframe_to_dataset(dataframe: pd.DataFrame):
    dataframe = dataframe.copy()
    labels = dataframe.pop("status")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds


def encode_numerical_feature(feature: K.Input, name: str, dataset: tf.data.Dataset):  # numerical features like age, salary etc
    # Create a Normalization layer for our feature
    normalizer = K.layers.Normalization()

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    normalizer.adapt(feature_ds)

    # Normalize the input feature
    encoded_feature = normalizer(feature)
    return encoded_feature


def encode_categorical_feature(feature: K.Input, name: str, dataset: tf.data.Dataset, is_string: bool):  # categorical features like sex, family status etc
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


def DataFilter(x, y):  # remove samples with 0 days before salary increase (as an experiment)
    return x['days_before_salary_increase'] > 0


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
    print("B", len(list(train_ds.as_numpy_iterator())))

    # train_ds = train_ds.filter(DataFilter)
    # val_ds = val_ds.filter(DataFilter)
    # val_ds_2 = val_ds.filter(DataFilter)


    print("A", len(list(train_ds.as_numpy_iterator())))

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