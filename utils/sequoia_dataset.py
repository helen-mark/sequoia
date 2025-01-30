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
        self.snapshot_duration = _data_config['basic']['snapshot_duration']

        self.data_dir = self.data_config['data_location']['data_path']
        self.filename = self.data_config['data_location']['file_name']


        # df2.merge(df1, how='union', on='ФИО')
        self.main_features = self.data_config['required_sheets']['basic']['features']
        self.common_features_name_mapping = {}
        self.specific_features = []

        # collect feature input names which are common for all snapshots:
        for f_name in self.main_features.keys():
            f = self.main_features[f_name]
            if 'kind' in f.keys() and f['kind'] == 'common':
                self.common_features_name_mapping[f['name']] = f['name_out']
            else:
                self.specific_features.append(f['name'])

        self.time_series_name_mapping = {}
        for key in self.data_config['required_sheets']:
            if 'kind' in self.data_config['required_sheets'][key].keys():
                if self.data_config['required_sheets'][key]['kind'] == 'time_series':
                    self.time_series_name_mapping[self.data_config['required_sheets'][key]['name']] = self.data_config['required_sheets'][key]['name_out']
        print(        self.time_series_name_mapping)

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

    def calc_time_since_latest_event(self, _event_dates: list, _snapshot_start: datetime.date):
        _event_dates.reverse()
        for date in _event_dates:
            if date < _snapshot_start:
                return _snapshot_start - date

    def check_leader_left(self):
        pass

    def check_has_meal(self):
        pass

    def check_has_insurance(self):
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

    def set_column_labels_as_dates(self, _input_df: pd.DataFrame):
        year = 2024
        new_columns = ['code']
        for i in range(1, 13):
            new_columns.append(date(year, i, 1))
        print((new_columns[3] - new_columns[2]).days)
        _input_df.columns = new_columns
        return _input_df

    def fill_salary(self, df_salary: pd.DataFrame, _dataset: pd.DataFrame, _snapshot_start: datetime.time):
        salary_longterm_col = pd.DataFrame({'code': [], 'salary_avg': []})
        salary_cur_col = pd.DataFrame({'code': [], 'salary_current': []})
        time_since_increase_col = pd.DataFrame({'code': [], 'time_since_salary_increase': []})
        count = 0

        for code in _dataset['code']:
            print('\npersonal code:', code)
            sample = df_salary.loc[df_salary['code'] == code]
            sample = sample.drop(columns='code')  # leave ony columns with salary values
            print(sample)

            salary_avg = self.calc_numerical_average(sample, 6, _snapshot_start)
            dates = self.calc_salary_increase_dates(sample)
            time_since_salary_increase = self.calc_time_since_latest_event(dates, _snapshot_start)
            print("Time since salary increase:", time_since_salary_increase)
            cur_salary = self.calc_salary_current(sample, _snapshot_start)

            salary_longterm_col.loc[count] = [code, salary_avg.item()]
            salary_cur_col.loc[count] = [code, cur_salary.item()]
            time_since_increase_col.loc[count] = [code, time_since_salary_increase]

            count += 1

        return salary_longterm_col, salary_cur_col, time_since_increase_col

    def fill_average_values(self, _dataset: pd.DataFrame, _feature_df: pd.DataFrame, _longterm_avg: pd.DataFrame, _shortterm_avg: pd.DataFrame, _snapshot_start: datetime.time):
        count = 0

        for code in _dataset['code']:
            print('\npersonal code:', code)
            sample = _feature_df.loc[_feature_df['code'] == code]
            sample = sample.drop(columns='code')  # leave ony columns with salary values
            print(sample)

            absent_avg_6m = self.calc_numerical_average(sample, 6, _snapshot_start)
            absent_avg_2m = self.calc_numerical_average(sample, 2, _snapshot_start)

            _longterm_avg.loc[count] = [code, absent_avg_6m.item()]
            _shortterm_avg.loc[count] = [code, absent_avg_2m.item()]

            count += 1

        return _shortterm_avg, _longterm_avg

    def calc_time_since_promotion(self, _input_file: str, _dataset: pd.DataFrame, _snapshot_start: datetime.time):
        sheet_name = 'Дата повышения'
        df = read_excel(_input_file, sheet_name=sheet_name)
        df = df.drop(columns='№')
        df.columns = ['code', 'Дата последнего повышения']

        count = 0
        days_since_prom = pd.DataFrame({'code': [], 'days_since_promotion' + '_shortterm': []})

        for code in _dataset['code']:
            sample = df.loc[df['code'] == code]
            sample = sample.drop(columns='code')  # leave ony columns with salary values
            date_str = str(sample.iloc[0].item())
            splt = date_str.split('.')
            date_dt = date(int(splt[2]), int(splt[1]), int(splt[0]))

            res = self.calc_time_since_latest_event([date_dt], _snapshot_start)
            days_since_prom.loc[count] = [code, res.days]
            count += 1
        return days_since_prom

    def process_timeseries(self, _input_file: os.path, _dataset: pd.DataFrame, _snapshot_start: datetime.time, _sheet_name: str, _feature_name: str):
        df = read_excel(_input_file, sheet_name=_sheet_name)

        df = df.drop(columns='№')
        df = self.set_column_labels_as_dates(df)

        if _feature_name in 'salary':
            return self.fill_salary(df, _dataset, _snapshot_start)
        longterm_col = pd.DataFrame({'code': [], _feature_name + '_longterm': []})
        shortterm_col = pd.DataFrame({'code': [], _feature_name + '_shortterm': []})

        return self.fill_average_values(_dataset, df, longterm_col, shortterm_col, _snapshot_start)

    def fill_snapshot_specific(self, _specific_features: list, _input_file: os.path, _dataset: pd.DataFrame, _snapshot_start: datetime.time):
        snapshot_columns = []

        for sheet_name, feature_name in self.time_series_name_mapping.items():
            snapshot_columns.extend(self.process_timeseries(_input_file, _dataset, _snapshot_start, sheet_name, feature_name))

        snapshot_columns.extend([self.calc_time_since_promotion(_input_file, _dataset, _snapshot_start)])

        for new_col in snapshot_columns:
            _dataset = _dataset.merge(new_col, on='code', how='outer')

        print(_dataset)

        # - age
        # - company_seniority
        # - overall_experience
        # - license_expiration
        # - income_avg
        # - income_current
        # - leader_left
        # - has_meal
        # - has_insurance

    def fill_common_features(self, _f_name, _dataset, _col):
        if _f_name == 'n':
            return
        reform_col = []
        for c in _col:
            f = self.lookup(_f_name, c)
            reform_col.append(f)
        _dataset.insert(len(_dataset.columns), _f_name, reform_col)

    def lookup(self, f_name, key):
        female = ['ж', 'Ж', 'жен', 'Жен', 'женский', 'Женский']
        male = ['м', 'М', 'муж', 'Муж', 'мужской', 'Мужской']
        russian = ['Россия', 'россия', 'Российское', 'российское']
        not_russian = ['иное', 'НЕ РФ', 'НЕ Рф']
        interm_education = ['среднее']
        high_education = ['высшее']
        spec_education = ['среднее специальное']
        married = ['женат', 'замужем', 'в браке']
        single = ['не женат', 'не замужем', 'не в браке']
        logistics = ['логистика']
        main_dept = ['основное производство', 'основной']

        if f_name == 'code':
            return int(key)
        elif f_name == 'gender':
            if key in female:
                return 1
            elif key in male:
                return 0
            else:
                raise ValueError(f'Invalid gender format: {key}')
        elif f_name == 'citizenship':
            if key in russian:
                return 0
            elif key in not_russian:
                return 1
            else:
                raise ValueError(f'Invalid citizenship format: {key}')
        elif f_name == 'education':
            if key in interm_education:
                return 0
            elif key in high_education:
                return 1
            elif key in spec_education:
                return 2
            else:
                raise ValueError(f'Invalid education format: {key}')
        elif f_name == 'family_status':
            if key in married:
                return 0
            elif key in single:
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
            if key in logistics:
                return 1
            elif key in main_dept:
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
        dataset = pd.DataFrame()

        for n, col in _input_df.transpose().iterrows():  # transpose dataframe to iterate over columns
            if col.name in _common_features:
                new_col = col.values
                self.fill_common_features(_common_features[col.name], dataset, new_col)

        print('common dataset result:', dataset)
        return dataset


    def check_and_parse(self):
        data_file_path = os.path.join(self.data_dir, self.filename)
        input_df_common = read_excel(data_file_path, sheet_name='Основные данные')

        dt1 = datetime.now()
        dt2 = datetime(2023, 1, 25)
        print(dt1.timestamp() - dt2.timestamp())

        main_dataset = self.collect_main_data(self.common_features_name_mapping, input_df_common, self.data_config)
        self.fill_snapshot_specific(self.specific_features, data_file_path, main_dataset, date(2024, 10, 2))


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
