from pandas import read_excel
import pandas as pd
import os
import yaml


def check_required_fields():
    pass


def calc_age():
    pass


def calc_company_seniority():
    pass


def calc_overall_experience():
    pass


def calc_license_expiration():
    pass


def calc_salary_6m_average():
    pass


def calc_salary_current():
    pass


def calc_time_since_salary_increase():
    pass


def calc_income_6m():
    pass


def calc_income_cur():
    pass


def calc_time_since_promotion():
    pass


def calc_absenteeism_6m():
    pass


def calc_absenteeism_2m():
    pass


def calc_vacation_days():
    pass


def check_leader_left():
    pass


def check_has_meal():
    pass


def check_has_insurance():
    pass


def calc_penalties_2m():
    pass


def calc_penalties_6m():
    pass


def fill_snapshot_specific():
    pass


def fill_common_features(_f_name, _dataset, _col):

    _dataset.insert(len(_dataset.columns), _f_name, _col)
    pass


def check_and_parse(_data_config, _dataset_config):
    data_dir = _data_config['data_location']['data_path']
    filename = _data_config['data_location']['file_name']

    input_df = read_excel(os.path.join(data_dir, filename), sheet_name='Основные данные')
    main_features = _data_config['required_sheets']['basic']['features']
    common_features = {}

    # collect feature input names which are common for all snapshots:
    for f_name in main_features.keys():
        f = main_features[f_name]
        if 'kind' in f.keys() and f['kind'] == 'common':
            common_features[f['name']] = f['name_out']
    print(f'common: {common_features}\n')

    dataset = pd.DataFrame()  # columns=[dataset_config['snapshot_features']["common"] + dataset_config['snapshot_features']["specific"]])

    print(dataset)
    new_col = []
    for n, row in input_df.transpose().iterrows():  # transpose dataframe to iterate over columns
        if row.name in common_features:
            new_col = row.values
            fill_common_features(common_features[row.name], dataset, new_col)
        else:
            pass
            # - age
            # - company_seniority
            # - overall_experience
            # - license_expiration
            # - salary_6m_average
            # - salary_current
            # - time_since_salary_increase
            # - income_6m_average
            # - income_current
            # - time_since_last_promotion
            # - absenteeism_6m_average
            # - absenteeism_2m_average
            # - vacation_days  # except last month?
            # - leader_left_3m
            # - has_meal
            # - has_insurance
            # - penalties_2m
            # - penalties_6m


    print('result:', dataset)

    pass


if __name__ == '__main__':
    setup_path = '../data_config.yaml'
    dataset_config_path = '../dataset_config.yaml'
    with open(setup_path) as stream:
        data_config = yaml.load(stream, yaml.Loader)
        # for key, val in data_config.items():
        #     print(val)

    with open(dataset_config_path) as stream:
        dataset_config = yaml.load(stream, yaml.Loader)
        # print(dataset_config['snapshot_features']["common"])

    check_and_parse(data_config, dataset_config)
