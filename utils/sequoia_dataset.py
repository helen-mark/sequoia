import os

from datetime import datetime, date
import pandas as pd
import numpy as np
import yaml
from pandas import read_excel

from path_setup import SequoiaPath


class SequoiaDataset:
    def __init__(self, _data_config, _dataset_config):
        self.data_config = _data_config
        self.dataset_config = _dataset_config

    def check_required_fields(self):
        pass

    def calc_age(self):
        pass

    def calc_company_seniority(self):
        pass

    def calc_overall_experience(self):
        pass

    def calc_license_expiration(self):
        pass

    def calc_numerical_average(self, _values_per_month: pd.DataFrame, _period_months: int, _snapshot_start: datetime.date):
        assert not _values_per_month.empty
        assert _period_months > 0

        _values_per_month = _values_per_month[_values_per_month.columns[::-1]]  # this operation reverses order of columns
        print('average salary: reversed sample = ', _values_per_month)

        values_sum = 0.
        count = 0
        for date, val in _values_per_month.items():  # same order of columns like in xlsx table
            if date <= _snapshot_start:
                print('Take salary at', date)
                values_sum += val
                count += 1
                if count == _period_months:
                    break

        avg = values_sum / _period_months
        print("Average:", avg)
        return avg

    def calc_salary_current(self, _salary_per_month: pd.DataFrame, _snapshot_start: datetime.date):
        assert len(_salary_per_month) > 0
        _salary_per_month = _salary_per_month[_salary_per_month.columns[::-1]]  # this operation reverses order of columns
        print('average salary: reversed sample = ', _salary_per_month)

        for date, val in _salary_per_month.items():  # same order of columns like in xlsx table
            if date <= _snapshot_start:
                print('Take salary at', date)
                return val

    def calc_time_since_salary_increase(self, _salary_increase_dates: list, _snapshot_start: datetime.date):
        _salary_increase_dates.reverse()
        for date in _salary_increase_dates:
            if date < _snapshot_start:
                return _snapshot_start - date

    def calc_time_since_promotion(self):
        pass

    def calc_vacation_days(self):
        pass

    def check_leader_left(self):
        pass

    def check_has_meal(self):
        pass

    def check_has_insurance(self):
        pass

    def calc_penalties(self, _period_months: int):
        pass

    def calc_salary_increase_dates(self, _salary_per_month: pd.DataFrame):
        dates = []
        assert not _salary_per_month.empty

        # months = ['Январь',	'Февраль',	'Март',	'Апрель',	'Май',	'Июнь',	'Июль',	'Август',	'Сентябрь',	'Октябрь',	'Ноябрь',	'Декабрь']
        salary_prev = 0
        for name, val in _salary_per_month.items():  # same order of columns like in xlsx table
            if val.item() > salary_prev:  # consider employers' first salary as increase too
                salary_prev = val.item()
                dates.append(name)
        return dates

    def fill_dates(self, _input_df: pd.DataFrame):
        year = 2024
        new_columns = ['code']
        for i in range(1, 13):
            new_columns.append(date(year, i, 1))
        print((new_columns[3] - new_columns[2]).days)
        _input_df.columns = new_columns
        return _input_df

    def fill_salary(self, _input_file: os.path, _dataset: pd.DataFrame, _snapshot_start: datetime.time):
        df_salary = read_excel(_input_file, sheet_name='Оплата труда')

        df_salary = df_salary.drop(columns='№')
        df_salary = self.fill_dates(df_salary)

        salary_longterm_col = pd.DataFrame({'code': [], 'salary_6m_average': []})
        salary_cur_col = pd.DataFrame({'code': [], 'salary_current': []})
        time_since_increase_col = pd.DataFrame({'code': [], 'time_since_salary_increase': []})
        count = 0

        for code in _dataset['code']:
            print('\npersonal code:', code)
            sample = df_salary.loc[df_salary['code'] == code]
            sample = sample.drop(columns='code')  # leave ony columns with salary values
            print(sample)

            salary_avg_6m = self.calc_numerical_average(sample, 6, _snapshot_start)
            dates = self.calc_salary_increase_dates(sample)
            time_since_salary_increase = self.calc_time_since_salary_increase(dates, _snapshot_start)
            print("Time since salary increase:", time_since_salary_increase)
            cur_salary = self.calc_salary_current(sample, _snapshot_start)

            salary_longterm_col.loc[count] = [code, salary_avg_6m.item()]
            salary_cur_col.loc[count] = [code, cur_salary.item()]
            time_since_increase_col.loc[count] = [code, time_since_salary_increase]

            count += 1

        return salary_longterm_col, salary_cur_col, time_since_increase_col

    def fill_absenteeism(self, _input_file: os.path, _dataset: pd.DataFrame, _snapshot_start: datetime.time):
        df_absent = read_excel(_input_file, sheet_name='Абсенцизм')

        df_absent = df_absent.drop(columns='№')
        df_absent = self.fill_dates(df_absent)
        absent_longterm_col = pd.DataFrame({'code': [], 'absenteeism_6m_average': []})
        absent_shortterm_col = pd.DataFrame({'code': [], 'absenteeism_2m_average': []})
        count = 0

        for code in _dataset['code']:
            print('\npersonal code:', code)
            sample = df_absent.loc[df_absent['code'] == code]
            sample = sample.drop(columns='code')  # leave ony columns with salary values
            print(sample)

            absent_avg_6m = self.calc_numerical_average(sample, 6, _snapshot_start)
            absent_avg_2m = self.calc_numerical_average(sample, 2, _snapshot_start)

            absent_longterm_col.loc[count] = [code, absent_avg_6m.item()]
            absent_shortterm_col.loc[count] = [code, absent_avg_2m.item()]

            count += 1

        return absent_shortterm_col, absent_longterm_col

    def fill_snapshot_specific(self, _specific_features: list, _input_file: os.path, _dataset: pd.DataFrame, _snapshot_start: datetime.time, _snapshot_dur: int):
        snapshot_columns = [0, 0, 0, 0, 0,]
        snapshot_columns[:3] = self.fill_salary(_input_file, _dataset, _snapshot_start)
        snapshot_columns[3:] = self.fill_absenteeism(_input_file, _dataset, _snapshot_start)

        for new_col in snapshot_columns:
            _dataset = _dataset.merge(new_col, on='code', how='outer')

        print(_dataset)

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

    def fill_common_features(self, _f_name, _dataset, _col):
        if _f_name == 'n':
            return
        reform_col = []
        for c in _col:
            f = self.lookup(_f_name, c)
            reform_col.append(f)
        _dataset.insert(len(_dataset.columns), _f_name, reform_col)

    def lookup(self, f_name, key):
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


    def collect_main_data(self, _common_features: {}, _input_df: pd.DataFrame, _data_config: dict):
        dataset = pd.DataFrame()  # columns=[dataset_config['snapshot_features']["common"] + dataset_config['snapshot_features']["specific"]])

        new_col = []
        for n, col in _input_df.transpose().iterrows():  # transpose dataframe to iterate over columns
            if col.name in _common_features:
                new_col = col.values
                self.fill_common_features(_common_features[col.name], dataset, new_col)

        print('result:', dataset)
        return dataset


    def check_and_parse(self):
        data_dir = self.data_config['data_location']['data_path']
        filename = self.data_config['data_location']['file_name']

        input_df_common = read_excel(os.path.join(data_dir, filename), sheet_name='Основные данные')

        # df2.merge(df1, how='union', on='ФИО')
        main_features = self.data_config['required_sheets']['basic']['features']
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

        main_dataset = self.collect_main_data(common_features, input_df_common, self.data_config)
        self.fill_snapshot_specific(specific_features, os.path.join(data_dir, filename), main_dataset, date(2024, 10, 2), 6)


if __name__ == '__main__':
    setup_path = '../data_config.yaml'
    dataset_config_path = '../dataset_config.yaml'
    with open(setup_path) as stream:
        data_config = yaml.load(stream, yaml.Loader)

    with open(dataset_config_path) as stream:
        dataset_config = yaml.load(stream, yaml.Loader)
        # print(dataset_config['snapshot_features']["common"])

    new_dataset = SequoiaDataset(data_config, dataset_config)
    new_dataset.check_and_parse()
