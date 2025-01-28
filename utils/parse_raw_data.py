import os

from datetime import datetime, date
import pandas as pd
import yaml
from pandas import read_excel

from path_setup import SequoiaPath

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


def calc_salary_average(_period_months: int):
    pass


def calc_salary_current():
    pass


def calc_time_since_salary_increase():
    pass


def calc_income(_period_months: int):
    pass


def calc_income_cur():
    pass


def calc_time_since_promotion():
    pass


def calc_absenteeism(_period_months: int):
    pass


def calc_vacation_days():
    pass


def check_leader_left():
    pass


def check_has_meal():
    pass


def check_has_insurance():
    pass


def calc_penalties(_period_months: int):
    pass



def fill_snapshot_specific(_snapshot_start: datetime.time):
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
    pass


def fill_common_features(_f_name, _dataset, _col):
    reform_col = []
    for c in _col:
        f = lookup(_f_name, c)
        reform_col.append(f)
    _dataset.insert(len(_dataset.columns), _f_name, reform_col)
    pass


def lookup(f_name, key):
    if f_name == 'n':
        return int(key)
    elif f_name == 'gender':
        if key in ['ж', 'Ж', 'жен', 'Жен', 'женский', 'Женский']:
            return 1
        elif key in ['м', 'М', 'муж', 'Муж', 'мужской', 'Мужской']:
            return 0
        else:
            raise ValueError(f'Invalid gender format: {key}')
    elif f_name == 'citizenship':
        if key in ['Россия', 'россия', 'Российское', 'российское']:
            return 0
        elif key in ['иное', 'НЕ РФ', 'НЕ Рф']:
            return 1
        else:
            raise ValueError(f'Invalid citizenship format: {key}')
    elif f_name == 'education':
        if key in ['среднее']:
            return 0
        elif key in ['высшее']:
            return 1
        elif key in ['среднее специальное']:
            return 2
        elif key in ['неоконченное высшее']:
            return 3
        else:
            raise ValueError(f'Invalid education format: {key}')

    elif f_name == 'family_status':
        if key in ['женат', 'замужем', 'в браке']:
            return 0
        elif key in ['не женат', 'не замужем', 'не в браке']:
            return 1
        else:
            raise ValueError(f'Invalid family status format: {key}')
    elif f_name == 'children':
        return int(key)
    elif f_name == 'to_work_travel_time':
        return float(key)
    elif f_name == 'department':
        if key in ['логистика']:
            return 1
        elif key in ['основное производство', 'основной']:
            return 0
        else:
            raise ValueError(f'Invalid department format: {key}')
    elif f_name == 'n_employers':
        return int(key)
    elif f_name == 'occupational_hazards':
        return int(key)
    else:
        raise ValueError(f'Invalid feature name passed to lookup(): {f_name}')


def check_and_parse(_data_config, _dataset_config):
    data_dir = _data_config['data_location']['data_path']
    filename = _data_config['data_location']['file_name']

    input_df = read_excel(os.path.join(data_dir, filename), sheet_name='Основные данные')
    main_features = _data_config['required_sheets']['basic']['features']
    common_features = {}
    specific_features = []

    # collect feature input names which are common for all snapshots:
    for f_name in main_features.keys():
        f = main_features[f_name]
        if 'kind' in f.keys() and f['kind'] == 'common':
            common_features[f['name']] = f['name_out']
        else:
            specific_features.append(f['name'])

    print(f'common: {common_features}\n')

    dataset = pd.DataFrame()  # columns=[dataset_config['snapshot_features']["common"] + dataset_config['snapshot_features']["specific"]])

    print(dataset)
    new_col = []
    for n, col in input_df.transpose().iterrows():  # transpose dataframe to iterate over columns
        if col.name in common_features:
            new_col = col.values
            print(new_col)
            fill_common_features(common_features[col.name], dataset, new_col)

    fill_snapshot_specific(specific_features)

    print('result:', dataset)


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
    dt1 = datetime.now()
    dt2 = datetime(2023, 1, 25)
    print(dt1.timestamp() - dt2.timestamp())

    check_and_parse(data_config, dataset_config)
