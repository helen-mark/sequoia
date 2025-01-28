import os

from datetime import datetime, date
import pandas as pd
import numpy as np
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


def calc_salary_average(_salary_per_month: pd.DataFrame, _period_months: int, _snapshot_start: datetime.date):
    assert not _salary_per_month.empty
    assert _period_months > 0

    _salary_per_month = _salary_per_month[_salary_per_month.columns[::-1]]  # this operation reverses order of columns
    print('average salary: reversed sample = ', _salary_per_month)

    salary_sum = 0.
    count = 0
    for date, val in _salary_per_month.items():  # same order of columns like in xlsx table
        if date <= _snapshot_start:
            print('Take salary at', date)
            salary_sum += val
            count += 1
            if count == _period_months:
                break

    salary_avg = salary_sum / _period_months
    print("Average salary:", salary_avg)
    return salary_avg


def calc_salary_current(_salary_per_month: pd.DataFrame, _snapshot_start: datetime.date):
    assert len(_salary_per_month) > 0
    _salary_per_month = _salary_per_month[_salary_per_month.columns[::-1]]  # this operation reverses order of columns
    print('average salary: reversed sample = ', _salary_per_month)

    for date, val in _salary_per_month.items():  # same order of columns like in xlsx table
        if date <= _snapshot_start:
            print('Take salary at', date)
            return val



def calc_time_since_salary_increase(_salary_increase_dates: list, _snapshot_start: datetime.date):
    _salary_increase_dates.reverse()
    for date in _salary_increase_dates:
        if date < _snapshot_start:
            return _snapshot_start - date


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


def calc_salary_increase_dates(_salary_per_month: pd.DataFrame):
    dates = []
    assert not _salary_per_month.empty

    # months = ['Январь',	'Февраль',	'Март',	'Апрель',	'Май',	'Июнь',	'Июль',	'Август',	'Сентябрь',	'Октябрь',	'Ноябрь',	'Декабрь']
    salary_prev = 0
    for name, val in _salary_per_month.items():  # same order of columns like in xlsx table
        if val.item() > salary_prev:  # consider employers' first salary as increase too
            salary_prev = val.item()
            dates.append(name)
    return dates


def fill_salary_per_date(_input_df: pd.DataFrame):
    year = 2024
    new_columns = ['code']
    for i in range(1, 13):
        new_columns.append(date(year, i, 1))
    print((new_columns[3] - new_columns[2]).days)
    _input_df.columns = new_columns
    return _input_df


def fill_snapshot_specific(_specific_features: list, _input_df: pd.DataFrame, _dataset: pd.DataFrame, _snapshot_start: datetime.time, _snapshot_dur: int):
    _input_df = _input_df.drop(columns='№')
    _input_df = fill_salary_per_date(_input_df)

    for code in _dataset['code']:
        print('\npersonal code:', code)
        sample = _input_df.loc[_input_df['code'] == code]
        sample = sample.drop(columns='code')  # leave ony columns with salary values
        print(sample)

        salary_avg_6m = calc_salary_average(sample, 6, _snapshot_start)
        print('after:', sample)
        dates = calc_salary_increase_dates(sample)
        time_since_salary_increase = calc_time_since_salary_increase(dates, _snapshot_start)
        print("Time since salary increase:", time_since_salary_increase)
        cur_salary = calc_salary_current(sample, _snapshot_start)

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


def fill_common_features(_f_name, _dataset, _col):
    if _f_name == 'n':
        return
    reform_col = []
    for c in _col:
        f = lookup(_f_name, c)
        reform_col.append(f)
    _dataset.insert(len(_dataset.columns), _f_name, reform_col)


def lookup(f_name, key):
    if f_name == 'code':
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
        #if not key.isdigit():
        #    raise TypeError(f'Unexpected n_children value: {key}')
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
        #if not key.isdigit():
        #    raise TypeError(f'Unexpected n_employers value: {key}')
        return int(key)
    elif f_name == 'occupational_hazards':
        #if not key.isdigit():
        #    raise TypeError(f'Unexpected hazards value: {key}')
        return int(key)
    else:
        raise ValueError(f'Invalid feature name passed to lookup(): {f_name}')


def collect_main_data(_common_features: {}, _input_df: pd.DataFrame, _data_config: dict):
    dataset = pd.DataFrame()  # columns=[dataset_config['snapshot_features']["common"] + dataset_config['snapshot_features']["specific"]])

    new_col = []
    for n, col in _input_df.transpose().iterrows():  # transpose dataframe to iterate over columns
        if col.name in _common_features:
            new_col = col.values
            fill_common_features(_common_features[col.name], dataset, new_col)

    print('result:', dataset)
    return dataset


def check_and_parse(_data_config, _dataset_config):
    data_dir = _data_config['data_location']['data_path']
    filename = _data_config['data_location']['file_name']

    input_df_common = read_excel(os.path.join(data_dir, filename), sheet_name='Основные данные')
    input_df_salary = read_excel(os.path.join(data_dir, filename), sheet_name='Оплата труда')

    # df2.merge(df1, how='union', on='ФИО')
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


    dt1 = datetime.now()
    dt2 = datetime(2023, 1, 25)
    print(dt1.timestamp() - dt2.timestamp())

    main_dataset = collect_main_data(common_features, input_df_common, _data_config)
    dataset = fill_snapshot_specific(specific_features, input_df_salary, main_dataset, date(2024, 10, 2), 6)



if __name__ == '__main__':
    setup_path = '../data_config.yaml'
    dataset_config_path = '../dataset_config.yaml'
    with open(setup_path) as stream:
        data_config = yaml.load(stream, yaml.Loader)

    with open(dataset_config_path) as stream:
        dataset_config = yaml.load(stream, yaml.Loader)
        # print(dataset_config['snapshot_features']["common"])

    check_and_parse(data_config, dataset_config)
