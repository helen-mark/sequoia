import tensorflow as tf
import tensorflow.keras as K
import pandas as pd
from pandas import read_csv
import numpy as np


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


def create_dataset(_dataset: pd.DataFrame, _dataset_val: pd.DataFrame, _batch_size: int):
    print(
        f"Using {len(_dataset)} samples for training "
        f"and {len(_dataset_val)} for validation"
    )

    train_ds = dataframe_to_dataset(_dataset)
    val_ds = dataframe_to_dataset(_dataset_val)
    val_ds_2 = dataframe_to_dataset(_dataset_val)
    print("B", len(list(train_ds.as_numpy_iterator())))

    # train_ds = train_ds.filter(DataFilter)
    # val_ds = val_ds.filter(DataFilter)
    # val_ds_2 = val_ds.filter(DataFilter)


    print("A", len(list(train_ds.as_numpy_iterator())))

    train_ds = train_ds.batch(_batch_size)
    val_ds = val_ds.batch(_batch_size)
    return train_ds, val_ds, val_ds_2, _dataset, _dataset_val


def prepare_all_features(_train_ds: tf.data.Dataset):
    # Categorical features encoded as integers
    gender = K.Input(shape=(1,), name="gender", dtype="int64")
    department = K.Input(shape=(1,), name="department", dtype="int64")
    nationality = K.Input(shape=(1,), name="citizenship", dtype="int64")
    field = K.Input(shape=(1,), name="field", dtype="int64")
    # family_status = K.Input(shape=(1,), name="family_status", dtype="int64")
    # children = K.Input(shape=(1,), name="children")
    # education = K.Input(shape=(1,), name="education")
    hazards = K.Input(shape=(1,), name="occupational_hazards", dtype="int64")

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
    field_encoded = encode_categorical_feature(field, "field", _train_ds, False)
    hazards_encoded = encode_categorical_feature(hazards, "occupational_hazards", _train_ds, False)
    #family_status_encoded = encode_categorical_feature(family_status, "family_status", _train_ds, False)
    #education_encoded = encode_categorical_feature(education, 'education', _train_ds, False)
    #children_encoded = encode_categorical_feature(children, 'children', _train_ds, False)

    # Numerical features
    age_encoded = encode_numerical_feature(age, "age", _train_ds)
    seniority_encoded = encode_numerical_feature(seniority, "seniority", _train_ds)
    vacation_days_s_encoded = encode_numerical_feature(vacation_days_s, "vacation_days_shortterm", _train_ds)
    #vacation_days_l_encoded = encode_numerical_feature(vacation_days_l, "vacation_days_longterm", _train_ds)
    external_factor_1_encoded = encode_numerical_feature(external_factor_1, "external_factor_1", _train_ds)
    external_factor_2_encoded = encode_numerical_feature(external_factor_2, "external_factor_2", _train_ds)
    external_factor_3_encoded = encode_numerical_feature(external_factor_3, "external_factor_3", _train_ds)

    # days_before_salary_increase_encoded = encode_numerical_feature(days_before_salary_increase,
    #                                                               "days_before_salary_increase", _train_ds)
    #salary_increase_encoded = encode_numerical_feature(salary_increase, "salary_increase", _train_ds)
    overtime_s_encoded = encode_numerical_feature(overtime_s, "overtime_shortterm", _train_ds)
    #overtime_l_encoded = encode_numerical_feature(overtime_l, "overtime_longterm", _train_ds)
    absenteeism_s_encoded = encode_numerical_feature(absenteeism_s, "absenteeism_shortterm", _train_ds)
    #absenteeism_l_encoded = encode_numerical_feature(absenteeism_l, "absenteeism_longterm", _train_ds)

    # km_to_work_encoded = encode_numerical_feature(km_to_work, "km_to_work", train_ds)
    #salary_6m_average_encoded = encode_numerical_feature(salary_6m_average, "income_longterm", _train_ds)
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
        #vacation_days_l,
        #days_before_salary_increase,
        #salary_increase,
        overtime_s,
        #overtime_l,
        absenteeism_s,
        #absenteeism_l,
       # family_status,
        # km_to_work,
        #salary_6m_average,
        salary_cur,
        field,
        hazards,
        # external_factor_1,
        # external_factor_2,
        # external_factor_3
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
            #vacation_days_l_encoded,
            overtime_s_encoded,
            #overtime_l_encoded,
            absenteeism_s_encoded,
            #absenteeism_l_encoded,
            #       family_status_encoded,
     #       children_encoded,
     #      education_encoded,
            # salary_6m_average_encoded,
            salary_cur_encoded,
            #days_before_salary_increase_encoded,
            #salary_increase_encoded,

            # km_to_work_encoded,
            field_encoded,
            hazards_encoded,
            # external_factor_1_encoded,
            # external_factor_2_encoded,
            # external_factor_3_encoded
        ]
    )
    all_features_reg = 0
    # K.layers.concatenate([
    #     external_factor_1_encoded,
    #     external_factor_2_encoded,
    #     external_factor_3_encoded])

    return all_features_reg, all_features_nonreg, all_inputs_reg, all_inputs_nonreg

def add_quality_features(df: pd.DataFrame):
    # df['age_group'] = pd.cut(df['age'], bins=[18, 25, 35, 45, 55, 65],
    #                         labels=['18-25', '26-35', '36-45', '46-55', '56-65'])
    # df['gender_age'] = df['gender'].astype(str) + '_' + df['age_group'].astype(str)
    df['citizenship_gender'] = df['citizenship'].astype(str) + '_' + df['gender'].astype(str)
    df['absences_per_experience'] = df['absenteeism_shortterm'] / (df['seniority'] + 1)
    df['unused_vacation_per_experience'] = df['vacation_days_shortterm'] / (df['seniority'] + 1)
    df['log_experience'] = np.log1p(df['seniority']+0.5)
    df['absences_per_year'] = df['absenteeism_shortterm'] / (df['seniority'] / 365 + 0.001)
    df['income_per_experience'] = df['income_shortterm'] / (df['seniority'] + 1)
    df['income_group'] = pd.qcut(df['income_shortterm'], q=5, labels=['low', 'medium_low', 'medium', 'medium_high', 'high'])
    df['position_industry'] = df['department'].astype(str) + '_' + df['field'].astype(str)
    # df['harm_position'] = df['occupational_hazards'].astype(str) + '_' + df['department'].astype(str)
    # df['harm_experience'] = df['occupational_hazards'].astype(str) + '_' + df['seniority'].astype(str)
    industry_avg_income = df.groupby('field')['income_shortterm'].mean().to_dict()
    df['industry_avg_income'] = df['field'].map(industry_avg_income)
    df['income_vs_industry'] = df['income_shortterm'] - df['industry_avg_income']
    position_median_income = df.groupby('department')['income_shortterm'].median().to_dict()
    df['position_median_income'] = df['department'].map(position_median_income)

    calculated_cat_feat = ['citizenship_gender', 'income_group', 'position_industry']
    return df, calculated_cat_feat

def create_features_for_datasets(_datasets: list):
    improved_datasets = []
    for d in _datasets:
        d, new_cat_feat = add_quality_features(d)
        cols = d.columns.tolist()
        print(cols)
        cols.remove('status')
        cols = cols + ['status']  # put 'status' to the end
        d = d[cols]
        improved_datasets.append(d)
    return improved_datasets, new_cat_feat