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
    nationality = K.Input(shape=(1,), name="citizenship", dtype="int64")
    # family_status = K.Input(shape=(1,), name="family_status", dtype="int64")
    # children = K.Input(shape=(1,), name="children")
    # education = K.Input(shape=(1,), name="education")

    # Numerical features
    age = K.Input(shape=(1,), name="age")
    seniority = K.Input(shape=(1,), name="seniority")
    vacation_days_s = K.Input(shape=(1,), name="vacation_days_shortterm")
    vacation_days_l = K.Input(shape=(1,), name="vacation_days_longterm")
    external_factor_1 = K.Input(shape=(1,), name="external_factor_1")
    external_factor_2 = K.Input(shape=(1,), name="external_factor_2")
    external_factor_3 = K.Input(shape=(1,), name="external_factor_3")
    # days_before_salary_increase = K.Input(shape=(1,), name="days_before_salary_increase")
    # salary_increase = K.Input(shape=(1,), name="salary_increase")
    overtime_s = K.Input(shape=(1,), name="overtime_shortterm")
    overtime_l = K.Input(shape=(1,), name="overtime_longterm")
    absenteeism_s = K.Input(shape=(1,), name="absenteeism_shortterm")
    absenteeism_l = K.Input(shape=(1,), name="absenteeism_longterm")

    # km_to_work = K.Input(shape=(1,), name="km_to_work")
    salary_6m_average = K.Input(shape=(1,), name="income_longterm")
    salary_cur = K.Input(shape=(1,), name="income_shortterm")


    # Integer categorical features
    gender_encoded = encode_categorical_feature(gender, "gender", _train_ds, False)
    department_encoded = encode_categorical_feature(department, "department", _train_ds, False)
    nationality_encoded = encode_categorical_feature(nationality, "citizenship", _train_ds, False)
    #family_status_encoded = encode_categorical_feature(family_status, "family_status", _train_ds, False)
    #education_encoded = encode_categorical_feature(education, 'education', _train_ds, False)
    #children_encoded = encode_categorical_feature(children, 'children', _train_ds, False)

    # Numerical features
    age_encoded = encode_numerical_feature(age, "age", _train_ds)
    seniority_encoded = encode_numerical_feature(seniority, "seniority", _train_ds)
    vacation_days_s_encoded = encode_numerical_feature(vacation_days_s, "vacation_days_shortterm", _train_ds)
    vacation_days_l_encoded = encode_numerical_feature(vacation_days_s, "vacation_days_longterm", _train_ds)
    external_factor_1_encoded = encode_numerical_feature(vacation_days_s, "external_factor_1", _train_ds)
    external_factor_2_encoded = encode_numerical_feature(vacation_days_s, "external_factor_2", _train_ds)
    external_factor_3_encoded = encode_numerical_feature(vacation_days_s, "external_factor_3", _train_ds)

    # days_before_salary_increase_encoded = encode_numerical_feature(days_before_salary_increase,
    #                                                               "days_before_salary_increase", _train_ds)
    #salary_increase_encoded = encode_numerical_feature(salary_increase, "salary_increase", _train_ds)
    overtime_s_encoded = encode_numerical_feature(overtime_s, "overtime_shortterm", _train_ds)
    overtime_l_encoded = encode_numerical_feature(overtime_l, "overtime_longterm", _train_ds)
    absenteeism_s_encoded = encode_numerical_feature(absenteeism_s, "absenteeism_shortterm", _train_ds)
    absenteeism_l_encoded = encode_numerical_feature(absenteeism_l, "absenteeism_longterm", _train_ds)

    # km_to_work_encoded = encode_numerical_feature(km_to_work, "km_to_work", train_ds)
    salary_6m_average_encoded = encode_numerical_feature(salary_6m_average, "income_longterm", _train_ds)
    salary_cur_encoded = encode_numerical_feature(salary_cur, "income_shortterm", _train_ds)


    all_inputs_nonreg = [
        department,
        seniority,
        nationality,
        age,
      #  children,
        gender,
      #  education,
        vacation_days_s,
        vacation_days_l,
        #days_before_salary_increase,
        #salary_increase,
        overtime_s,
        overtime_l,
        absenteeism_s,
        absenteeism_l,
       # family_status,
        # km_to_work,
        salary_6m_average,
        salary_cur,
        external_factor_1,
        external_factor_2,
        external_factor_3
    ]

    all_inputs_reg = [
        external_factor_1,
        external_factor_2,
        external_factor_3
    ]

    all_features_nonreg = K.layers.concatenate(
        [
            department_encoded,
            seniority_encoded,
            nationality_encoded,
            age_encoded,
            gender_encoded,
            vacation_days_s_encoded,
            vacation_days_l_encoded,
            overtime_s_encoded,
            overtime_l_encoded,
            absenteeism_s_encoded,
            absenteeism_l_encoded,
            #       family_status_encoded,
     #       children_encoded,
     #      education_encoded,
            salary_6m_average_encoded,
            salary_cur_encoded,
            #days_before_salary_increase_encoded,
            #salary_increase_encoded,

            # km_to_work_encoded,
            external_factor_1_encoded,
            external_factor_2_encoded,
            external_factor_3_encoded
        ]
    )
    all_features_reg = K.layers.concatenate([
        external_factor_1_encoded,
        external_factor_2_encoded,
        external_factor_3_encoded]
    )

    return all_features_reg, all_features_nonreg, all_inputs_reg, all_inputs_nonreg