import tensorflow as tf
import tensorflow.keras as K
from sklearn.preprocessing import OneHotEncoder

import pandas as pd
import os
import numpy as np
from imblearn.over_sampling import SMOTENC

# from ydata.connectors import LocalConnector
# from ydata.metadata import Metadata
# from ydata.synthesizers.regular.model import RegularSynthesizer
# from ydata.dataset.dataset import Dataset
# from ydata.report import SyntheticDataProfile
from sklearn.utils import shuffle

from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer, CopulaGANSynthesizer


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


def minority_class_resample(_datasets: list, _cat_feat: list):
    new_datasets = []
    for d in _datasets:
        # Get categorical feature indices dynamically
        categorical_indices = [d.columns.get_loc(col) for col in _cat_feat]

        # Initialize SMOTENC with auto-detected categorical indices
        smote = SMOTENC(
            categorical_features=categorical_indices,
            sampling_strategy=0.25,
            k_neighbors=5,
            random_state=42
        )

        X = d.drop('status', axis=1)
        y = d['status']

        # Check for NaN values (optional)
        for n, row in X.iterrows():
            if row.isnull().values.any():
                print(f"NaN value in snapshot dataset: {row}")

        # Resample
        X_resampled, y_resampled = smote.fit_resample(X, y)
        df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        df_resampled['status'] = y_resampled
        new_datasets.append(df_resampled)

    return new_datasets

def remove_short_service(_datasets: list):
    new_datasets = []
    for d in _datasets:
        d = d.loc[d['seniority'] >= 0.3].reset_index(drop=True)
        #d = d.rename(columns={'department': 'job_category'})
        new_datasets.append(d)
    return new_datasets

def collect_datasets(_data_path: str, _remove_small_period: bool = False, _remove_invalid_deriv: bool = False):
    datasets = []
    for filename in sorted(os.listdir(_data_path)):
        if '.csv' not in filename:
            continue
        dataset_path = os.path.join(_data_path, filename)
        print(filename)
        dataset = pd.read_csv(dataset_path)
        print("cols before", dataset.columns)

        if _remove_small_period:
            dataset['termination_date'] = pd.to_datetime(dataset['termination_date'])
            dataset = dataset[~((dataset['termination_date'].dt.month <= 2) & (dataset['termination_date'].dt.year == 2024))]
        if _remove_invalid_deriv:
            for col in dataset.columns:
                if 'deriv' in col:
                    dataset = dataset[dataset[col] != -1000000]
                    print('removed wrong derivative')

        strings_to_drop = ['long', 'birth', 'external_factor_', 'recruit', 'code', 'date', 'depar',  'hazards', 'Unnamed', 'internal', 'deriv']
                           #'deriv', 'internal', 'family', 'children', 'overtime', 'education', 'region', 'city', 'total', 'vacations_per', 'salary_per']  #'overtime'
        dataset = dataset.drop(
            columns=[c for c in dataset.columns if any(string in c for string in strings_to_drop)])
        print("cols after", dataset.columns)
        dataset = shuffle(dataset)

        my_list = ['инженер', 'производство', 'производство высококвалифицированное', 'управление', 'водитель', 'офис', 'сотрудник склада', 'медицинский персонал', 'логистика']  # замените на ваш реальный список

        # Функция для преобразования значений
        def transform_category(value):
            if value in my_list:
                return value  # оставляем как есть, если в списке
            elif 'водитель' in str(value).lower():  # проверяем наличие "водитель" (регистронезависимо)
                return 'водитель'
            elif 'склад' in str(value).lower():
                return 'сотрудник склада'
            else:
                return 'производство'

        # Применяем преобразование к столбцу
        dataset['job_category'] = dataset['job_category'].apply(transform_category)
        datasets.append(dataset)

    return datasets


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

def make_synthetic(_dataset: pd.DataFrame, _size: int, _type: str = 'sdv'):
    if _type == 'sdv':
        return make_sdv(_dataset, _size)
    else:
        return make_ydata(_dataset, _size)

def make_ydata(_dataset: pd.DataFrame, _size: int = 1000):
    # connector = LocalConnector()  # , keyfile_dict=token)
    #
    # # Instantiate a synthesizer
    # cardio_synth = RegularSynthesizer()
    # # memory_usage = trn.memory_usage(index=True, deep=False).sum()
    # # npartitions = int(((memory_usage / 10e5) // 100) + 1)
    # data = Dataset(_dataset)
    # # calculating the metadata
    # metadata = Metadata(data)
    #
    # # fit model to the provided data
    # cardio_synth.fit(data, metadata, condition_on=["status"])
    #
    # # Generate data samples by the end of the synth process
    # synth_sample = cardio_synth.sample(n_samples=_size,
    #                                    # condition_on={
    #                                    #  "status": {
    #                                    #      "categories": [{
    #                                    #          "category": 0,
    #                                    #          "percentage": 1.0
    #                                    #      }]
    #                                    #  }}
    #                                    )
    #
    # # TODO target variable validation
    # profile = SyntheticDataProfile(
    #     data,
    #     synth_sample,
    #     metadata=metadata,
    #     target="status",
    #     data_types=cardio_synth.data_types)
    #
    # profile.generate_report(
    #     output_path="./cardio_report_example.pdf",
    # )
    # return synth_sample.to_pandas()  # {t: df.to_pandas() for t, df in res.items()}

    pass

def make_sdv(_dataset: pd.DataFrame, _size: int = 1000):
    metadata = Metadata.detect_from_dataframe(
        data=_dataset,
        table_name='attrition')
    synthesizer = GaussianCopulaSynthesizer(metadata=metadata,
                                            numerical_distributions={
                                                'income_shortterm': 'gamma',
                                                'age': 'truncnorm',
                                                'absenteeism_shortterm': 'gamma',
                                                'seniority': 'norm',
                                                'vacation_days_shortterm': 'gamma'
                                            })

    df = _dataset.copy()
    synthesizer = CTGANSynthesizer(metadata=metadata,
                                   epochs=100,  # Increase from default 300 if needed
                                   batch_size=100,  # (len(df) // 4) // 10 * 10 ,  # ~25% of dataset size
                                   generator_dim=(128, 128),  # Larger networks for complex relationships
                                   discriminator_dim=(128, 128),
                                   verbose=True,  # To monitor training progress
                                   pac=5,  # Helps with mode collapse for categoricals
                                   cuda=True
                                   )

    synthesizer = TVAESynthesizer(metadata=metadata)
    for col in df.select_dtypes(include=['category']).columns:
        df[col] = df[col].astype('object')
    synthesizer.fit(df)

    synthetic_data = synthesizer.sample(num_rows=_size)
    return synthetic_data


def encode_categorical(_dataset: pd.DataFrame, _encoder: OneHotEncoder, _cat_features: list):
    encoded_features = _encoder.transform(_dataset[_cat_features]).toarray().astype(int)
    encoded_df = pd.DataFrame(encoded_features, columns=_encoder.get_feature_names_out(_cat_features))
    numerical_part = _dataset.drop(columns=_cat_features)

    return pd.concat([encoded_df, numerical_part], axis=1), encoded_df


def get_united_dataset(_d_train: list, _d_val: list, _d_test: list):
    trn = pd.concat(_d_train, axis=0)
    vl = pd.concat(_d_val, axis=0)

    x_train = trn.drop(columns=['status'])
    x_val = vl.drop(columns=['status'])
    y_train = trn['status']
    y_val = vl['status']
    return x_train, y_train, x_val, y_val

def get_split(_dataset: pd.DataFrame, _test_split: float, _split_state: int):
    # val = _dataset.sample(frac=_test_split, random_state=_split_state)
    # test = val  # .sample(frac=0.3, random_state=_split_rand_state)
    # # val = val.drop(test.index)
    # trn = _dataset.drop(val.index)

    n_splits = 5
    fold_size = len(_dataset) // n_splits

    start_index = _split_state * fold_size
    end_index = (_split_state + 1) * fold_size if _split_state < n_splits - 1 else len(_dataset)

    trn = pd.concat([_dataset.iloc[:start_index], _dataset.iloc[end_index:]])
    val = _dataset.iloc[start_index:end_index]
    test = val

    return trn, val, test

def prepare_dataset_2(_datasets: list, _test_datasets: list, _make_synthetic: bool, _encode_categorical: bool, _feature_selection: bool, _test_split: float, _cat_feat: list, _split_rand_state: int):
    cat_feature_names = _cat_feat
    if _encode_categorical:
        concat_dataset = pd.concat(_datasets+_test_datasets)
        encoder = OneHotEncoder()
        encoder.fit(concat_dataset[_cat_feat])

    # if _make_synthetic is not None:
    #     united = pd.concat(_datasets, axis=0)
    #     sample_df = make_synthetic(united,1000)
    #
    #     # Check for similar rows and print if match:
    #     for n, syn_row in united.iterrows():
    #         for n1, real_row in sample_df.iterrows():
    #             if (syn_row.values.astype(int) == real_row.values.astype(int)).all():
    #                 print("Match!", real_row.values.astype(int), syn_row.values.astype(int))
    #
    #     if _encode_categorical:
    #         sample_df, _ = encode_categorical(sample_df, encoder)

    if _make_synthetic is not None:
      trns = []
      for d in _datasets:
          trn, val, test = get_split(d, _test_split, _split_rand_state)
          trns.append(trn)
      d = pd.concat(trns, axis=0)
      sample_trn = make_synthetic(d, len(d)//3)
      if _encode_categorical:
          sample_trn, _ = encode_categorical(sample_trn, encoder, _cat_feat)

    n = 0
    d_val = []
    d_train = []
    d_test = []
    selected_feat = ['status', 'education_1', 'education_2', 'education_3', 'family_status_0',
       'family_status_1', 'family_status_2', 'gender_age_0_26-35',
       'gender_age_0_36-45', 'gender_age_0_46-55', 'gender_age_0_56-65',
       'citizenship_gender_0_0', 'citizenship_gender_0_1', 'age_group_18-25',
       'age_group_26-35', 'age_group_36-45', 'age_group_46-55',
       'absenteeism_shortterm', 'age', 'children', 'income_shortterm',
       'overtime_shortterm', 'total_seniority', 'vacation_days_shortterm',
       'absences_per_experience', 'unused_vacation_per_experience',
       'log_seniority', 'absences_per_year', 'income_per_experience',
       'income_vs_industry', 'position_median_income', 'income_vs_position',
       'age_sqr',
                     'salary_by_city', 'salary_vs_city', 'vacations_by_city',
                     'absenteeism_deriv', 'overtime_deriv', 'vacation_days_deriv']
    for dataset in _datasets + _test_datasets:

        n += 1
        # if _make_synthetic:
        #     sample_df = make_synthetic(dataset)
        # if _make_synthetic is not None:
        #     trn, val, test = get_split(dataset, _test_split, _split_rand_state)
        #
        #     sample_trn = make_synthetic(trn, int(len(trn)/3))

            # Check for similar rows and print if match:
            # for n, syn_row in sample_trn.iterrows():
            #     for n1, real_row in trn.iterrows():
            #         cond1 = syn_row.values.astype(float) * 0.99 < real_row.values.astype(float)
            #         cond2 = syn_row.values.astype(float) * 1.01 > real_row.values.astype(float)
            #         if (cond1 & cond2).all():
            #             print("Match!", real_row.values.astype(int), syn_row.values.astype(int))

        if _encode_categorical:
            # Convert the encoded features to a DataFrame
            dataset, encoded_part = encode_categorical(dataset, encoder, _cat_feat)
            # if _make_synthetic is not None:
            #     sample_trn, _ = encode_categorical(sample_trn, encoder, _cat_feat)
            cat_feature_names = encoded_part.columns.values.tolist()

        if _feature_selection:
            dataset = dataset.drop(columns = [c for c in dataset.columns if c not in selected_feat])
            cat_feature_names = [x for x in cat_feature_names if x in selected_feat]
        if n <= len(_datasets):
            d_train.append(dataset)
        else:
            d_val.append(dataset)

        # trn, val, test = get_split(dataset, _test_split, _split_rand_state)
        # #
        # # if _make_synthetic is not None:
        # #     trn = pd.concat([trn, sample_trn], axis=0)
        #
        # d_val.append(val)
        # d_train.append(trn)
        # d_test.append(test)
    print(334, len(d_val[0]))


    if _make_synthetic:
        d_train[0] = pd.concat([d_train[0], sample_trn], axis=0)
    return d_train, d_val, d_val, cat_feature_names


def prepare_all_features(_train_ds: tf.data.Dataset):
    # Categorical features encoded as integers
    gender = K.Input(shape=(1,), name="gender", dtype="int64")
    job_category = K.Input(shape=(1,), name="job_category", dtype="int64")
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
    job_category_encoded = encode_categorical_feature(job_category, "job_category", _train_ds, False)
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
        job_category,
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
            job_category_encoded,
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

def add_quality_features(df: pd.DataFrame, _total_ds: pd.DataFrame):
    original_types = df.dtypes

    df['age_group'] = pd.cut(df['age'], bins=[18, 25, 35, 45, 55, 65],
                             labels=['18-25', '26-35', '36-45', '46-55', '56-65'])
    df['gender_age'] = df['gender'].astype(str) + '_' + df['age_group'].astype(str)


    df['citizenship_gender'] = df['citizenship'].astype(str) + '_' + df['gender'].astype(str)
    #for name, count in df['job_category'].value_counts().items():
    #    print(f"{name}: {count}")
    df['job_category'] = np.where(df['job_category'].map(df['job_category'].value_counts()) < 50, 'Other',
                                  df['job_category'])
    if 'industry_kind_internal' in df.columns:
        df['industry_kind_internal'] = np.where(df['industry_kind_internal'].map(df['industry_kind_internal'].value_counts()) < 50, 'Other',
                                  df['industry_kind_internal'])

    df['absences_per_experience'] = df['absenteeism_shortterm'] / (df['seniority'] + 1)

    if 'vacation_days_shortterm' in df.columns:
        df['unused_vacation_per_experience'] = df['vacation_days_shortterm'] / (df['seniority'] + 1)
    df['log_seniority'] = np.log1p(df['seniority']+0.5)
    df['absences_per_year'] = df['absenteeism_shortterm'] / (df['seniority'] / 365 + 0.001)
    df['income_per_experience'] = df['income_shortterm'] / (df['seniority'] + 1)
    quantiles = pd.qcut(_total_ds['income_shortterm'], q=5, retbins=True)[1]

    # Then apply these same boundaries to your subset dataframe
    df['income_group'] = pd.cut(
        df['income_shortterm'],
        bins=quantiles,
        labels=['low', 'medium_low', 'medium', 'medium_high', 'high'][:len(quantiles)-1],
        include_lowest=True
    )


    if 'education' in df.columns and 'income_group' in df.columns:
        df['education_income'] = df['education'].astype(str) + '_' + df['income_group'].astype(str)

    # for col in df.columns:
    #     if 'deriv' in col:
    #         quantiles = pd.qcut(_total_ds[col], q=5, retbins=True, duplicates='drop')[1]
    #
    #         df[col+'_cat'] = pd.cut(df[col],
    #         bins=quantiles,
    #         labels=['large_neg', 'neg', 'zero', 'pos', 'large_pos'][:len(quantiles)-1],
    #         duplicates='drop', include_lowest=True)
    #
    #         df = df.drop(columns=[col])

    # df['income_group'] = pd.qcut(_total_ds['income_shortterm'], q=5, labels=['low', 'medium_low', 'medium', 'medium_high', 'high'])
    # quantiles = pd.qcut(_total_ds['city_population'], q=5, retbins=True, duplicates='drop')[1]
    # df['region_population_group'] = pd.cut(
    #     df['city_population'],
    #     bins=quantiles,
    #     labels=['low', 'medium_low', 'medium', 'medium_high', 'high'][:len(quantiles)-1],
    #     include_lowest=True
    # )
    #df['region_population_group'] = pd.qcut(_total_ds['region'], q=5, labels=['low', 'medium_low', 'medium', 'medium_high', 'high'], duplicates='drop')

    df['position_industry'] = df['job_category'].astype(str) + '_' + df['field'].astype(str)
    # df['harm_position'] = df['occupational_hazards'].astype(str) + '_' + df['job_category'].astype(str)
    # df['harm_experience'] = df['occupational_hazards'].astype(str) + '_' + df['seniority'].astype(str)
    industry_avg_income = _total_ds.groupby('field')['income_shortterm'].mean().to_dict()
    df['industry_avg_income'] = df['field'].map(industry_avg_income)
    df['income_vs_industry'] = df['income_shortterm'] / df['industry_avg_income']

    df['salary_by_city'] = np.where(
        df['salary_by_city'].isna() | (df['salary_by_city'] == 0),
        90000,  # Default value when divisor is invalid
        df['salary_by_city']
    )
    #df = df.drop(columns=['city_population', 'salary_by_city', 'seniority'], errors='ignore')
    if 'city' in df.columns:
        df['city'] = df['city'].where(df['city'].isin(['Невинномысск', 'Кингисепп', 'Белореченск', 'Новомосковск']), 'Other')
    df['salary_vs_city'] = np.where(
        df['salary_by_city'].isna() | (df['salary_by_city'] == 0),
        1,  # Default value when divisor is invalid
        df['income_shortterm'] / df['salary_by_city']
    )
    df = df.drop(columns=['salary_by_city'])
    position_median_income = df.groupby('job_category')['income_shortterm'].median().to_dict()
    df['position_median_income'] = df['job_category'].map(position_median_income)
    df['income_vs_position'] = df['income_shortterm'] / df['position_median_income']
    df['age_sqr'] = df['age'] ** 2.

    new_columns = set(df.columns) - set(original_types.index)
    # Filter for categorical (str/object) columns only
    calculated_cat_feat = [
        col for col in new_columns
        if pd.api.types.is_string_dtype(df[col])  # Checks if dtype is string/object
    ]
    #calculated_cat_feat = ['income_group', 'position_industry', 'age_group', 'gender_age', 'citizenship_gender', 'education_income', 'absenteeism_deriv', 'overtime_deriv', 'vacation_days_deriv', 'income_deriv']
    return df, calculated_cat_feat

def create_features_for_datasets(_datasets: list):
    improved_datasets = []
    tot_ds = pd.concat(_datasets, axis=0)
    print("TOT\n", tot_ds)
    for d in _datasets:
        d, new_cat_feat = add_quality_features(d, tot_ds)
        cols = d.columns.tolist()
        cols.remove('status')
        cols = cols + ['status']  # put 'status' to the end
        d = d[cols]
        improved_datasets.append(d)
    return improved_datasets, new_cat_feat