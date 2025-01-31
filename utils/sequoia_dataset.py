import os

from datetime import datetime, date
import pandas as pd
import numpy as np
import yaml
from pandas import read_excel

import logging
logging.basicConfig(level=logging.DEBUG)


class SnapShot:
    def __init__(self, _dur: int, _initial_offset: int, _snapshot_num: int):
        self.duration = _dur  # amount of days
        self.min_duration = 3
        self.initial_offset = _initial_offset  # amount of days
        self.num = _snapshot_num

    def total_offset(self):
        return self.initial_offset + self.duration*self.num


class SequoiaDataset:
    def __init__(self, _data_config, _dataset_config):
        logging.info("Initializing new dataset instance...")
        self.data_config = _data_config
        self.dataset_config = _dataset_config
        self.snapshot_duration = _data_config['basic']['snapshot_duration']

        self.data_dir = self.data_config['data_location']['data_path']
        self.filename = self.data_config['data_location']['file_name']
        self.output_path = os.path.join(self.data_dir, self.data_config['data_location']['output_name'])

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

        values_sum = 0.
        count = 0
        for date, val in _values_per_month.items():  # same order of columns like in xlsx table
            if date <= _snapshot_start:
                values_sum += val
                count += 1
                if count == _period_months:
                    break
        if count == 0:
            return None
        avg = values_sum / count
        return avg

    def calc_salary_current(self, _salary_per_month: pd.DataFrame, _snapshot_start: datetime.date):
        assert len(_salary_per_month) > 0
        _salary_per_month = _salary_per_month[_salary_per_month.columns[::-1]]  # this operation reverses order of columns

        for date, val in _salary_per_month.items():  # same order of columns like in xlsx table
            if date <= _snapshot_start:
                return val
        return None

    def calc_time_since_latest_event(self, _event_dates: list, _snapshot_start: datetime.date):
        _event_dates.reverse()
        for date in _event_dates:
            if date < _snapshot_start:
                return _snapshot_start - date
        return None

    def check_leader_left(self):
        pass

    def check_has_meal(self):
        pass

    def check_has_insurance(self):
        pass

    def calc_salary_increase_dates(self, _salary_per_month: pd.DataFrame):
        dates = []
        assert not _salary_per_month.empty

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
        _input_df.columns = new_columns
        return _input_df

    def calc_individual_snapshot_start(self, _snapshot: SnapShot, _recruitment_date: date, _term_date: date):
        offset_months = _snapshot.num * _snapshot.duration + _snapshot.initial_offset
        offset_years = int(np.floor(offset_months / 12))
        offset_months = offset_months % 12
        year = _term_date.year - offset_years
        month = _term_date.month - offset_months
        if month <= 0:
            month = 12 + month
            year -= 1
        if year < _recruitment_date.year or (year == _recruitment_date.year and month < _recruitment_date.month + 2):
            return None
        snapshot_start = date(year, month, 1)
        return snapshot_start


    def fill_salary(self, df_salary: pd.DataFrame, _dataset: pd.DataFrame, _snapshot: SnapShot):
        salary_longterm_col = pd.DataFrame({'code': [], 'salary_avg': []})
        salary_cur_col = pd.DataFrame({'code': [], 'salary_current': []})
        time_since_increase_col = pd.DataFrame({'code': [], 'time_since_salary_increase': []})
        count = 0

        for code in _dataset['code']:
            sample = df_salary.loc[df_salary['code'] == code]
            sample = sample.drop(columns='code')  # leave ony columns with salary values
            person_recruitment_date = _dataset.loc[_dataset['code'] == code]['recruitment_date'].item()
            person_termination_date = _dataset.loc[_dataset['code'] == code]['termination_date'].item()

            snapshot_start = self.calc_individual_snapshot_start(_snapshot, person_recruitment_date, person_termination_date)

            salary_avg = self.calc_numerical_average(sample, 6, snapshot_start)
            dates = self.calc_salary_increase_dates(sample)
            time_since_salary_increase = self.calc_time_since_latest_event(dates, snapshot_start)
            cur_salary = self.calc_salary_current(sample, snapshot_start)

            salary_longterm_col.loc[count] = [code, salary_avg.item()]
            salary_cur_col.loc[count] = [code, cur_salary.item()]
            time_since_increase_col.loc[count] = [code, time_since_salary_increase]

            count += 1

        return salary_longterm_col, salary_cur_col, time_since_increase_col

    def fill_average_values(self, _dataset: pd.DataFrame, _feature_df: pd.DataFrame, _longterm_avg: pd.DataFrame, _shortterm_avg: pd.DataFrame, _snapshot: SnapShot):
        count = 0

        for code in _dataset['code']:
            sample = _feature_df.loc[_feature_df['code'] == code]
            sample = sample.drop(columns='code')  # leave ony columns with salary values
            person_recruitment_date = _dataset.loc[_dataset['code'] == code]['recruitment_date'].item()
            person_termination_date = _dataset.loc[_dataset['code'] == code]['termination_date'].item()

            snapshot_start = self.calc_individual_snapshot_start(_snapshot, person_recruitment_date,
                                                                 person_termination_date)

            avg_6m = self.calc_numerical_average(sample, 6, snapshot_start)
            avg_2m = self.calc_numerical_average(sample, 2, snapshot_start)
            _longterm_avg.loc[count] = [code, avg_6m.item()]
            _shortterm_avg.loc[count] = [code, avg_2m.item()]

            count += 1

        return _shortterm_avg, _longterm_avg

    def apply_snapshot_specific_codes(self, _dataset: pd.DataFrame, _snapshot_num: int):
        codes = _dataset['code']
        new_codes = []
        for code in codes:
            new_codes.append(str(code) + '_s' + str(_snapshot_num))
        _dataset['code'] = new_codes
        return _dataset

    def str_to_datetime(self, _date_str: str):
        splt = _date_str.split('.')
        return date(int(splt[2]), int(splt[1]), int(splt[0]))

    def calc_time_since_promotion(self, _input_file: str, _dataset: pd.DataFrame, _snapshot: SnapShot):
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
            date_dt = self.str_to_datetime(date_str)
            person_recruitment_date = _dataset.loc[_dataset['code'] == code]['recruitment_date'].item()
            person_termination_date = _dataset.loc[_dataset['code'] == code]['termination_date'].item()

            snapshot_start = self.calc_individual_snapshot_start(_snapshot, person_recruitment_date,
                                                                 person_termination_date)

            res = self.calc_time_since_latest_event([date_dt], snapshot_start)
            days_since_prom.loc[count] = [code, res.days]
            count += 1
        return days_since_prom

    def process_timeseries(self, _input_file: os.path, _dataset: pd.DataFrame, _snapshot: SnapShot, _sheet_name: str, _feature_name: str):
        df = read_excel(_input_file, sheet_name=_sheet_name)

        df = df.drop(columns='№')
        df = self.set_column_labels_as_dates(df)

        if _feature_name == 'salary':
            return self.fill_salary(df, _dataset, _snapshot)
        longterm_col = pd.DataFrame({'code': [], _feature_name + '_longterm': []})
        shortterm_col = pd.DataFrame({'code': [], _feature_name + '_shortterm': []})
        return self.fill_average_values(_dataset, df, longterm_col, shortterm_col, _snapshot)

    def fill_snapshot_specific(self, _specific_features: list, _input_file: os.path, _dataset: pd.DataFrame, _snapshot: SnapShot):
        snapshot_columns = []

        for sheet_name, feature_name in self.time_series_name_mapping.items():
            snapshot_columns.extend(self.process_timeseries(_input_file, _dataset, _snapshot, sheet_name, feature_name))

        snapshot_columns.extend([self.calc_time_since_promotion(_input_file, _dataset, _snapshot)])

        for new_col in snapshot_columns:
            _dataset = _dataset.merge(new_col, on='code', how='outer')

        _dataset = self.apply_snapshot_specific_codes(_dataset, _snapshot.num)

        return _dataset

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
        elif f_name in ['recruitment_date', 'termination_date']:
            return self.str_to_datetime(str(key))
        else:
            raise ValueError(f'Invalid feature name passed to lookup(): {f_name}')

    def collect_main_data(self, _common_features: {}, _input_df: pd.DataFrame, _data_config: dict, _snapshot: SnapShot):
        dataset = pd.DataFrame()

        for n, row in _input_df.iterrows():
            recr_date = row['Дата найма']
            term_date = row['Дата увольнения']
            empl_period = (self.str_to_datetime(term_date)-self.str_to_datetime(recr_date)).days / 30
            # remove sample if it has no data for current time snapshot:
            if empl_period - _snapshot.min_duration < _snapshot.total_offset():
                _input_df = _input_df.drop(axis='index', labels=n)

        if _input_df.empty:
            return None

        for n, col in _input_df.transpose().iterrows():  # transpose dataframe to iterate over columns
            if col.name in _common_features:
                new_col = col.values
                self.fill_common_features(_common_features[col.name], dataset, new_col)

        return dataset

    def run(self):
        data_file_path = os.path.join(self.data_dir, self.filename)
        input_df_common = read_excel(data_file_path, sheet_name='Основные данные')

        snapshot_dur = 6
        snapshot = SnapShot(snapshot_dur, 3, 0)
        united_dataset = pd.DataFrame()
        while True:
            logging.info(f'Starting snapshot {snapshot.num}')
            main_dataset = self.collect_main_data(self.common_features_name_mapping, input_df_common, self.data_config, snapshot)
            if main_dataset is None:
                logging.info(f'No data left for snapshot {snapshot.num}. Finishing process...')
                break
            logging.debug(f'Formed main part of dataset: {main_dataset}')
            full_snapshot_dataset = self.fill_snapshot_specific(self.specific_features, data_file_path, main_dataset, snapshot)
            logging.debug(f'Final dataset for snapshot {snapshot.num}: {full_snapshot_dataset}')
            if snapshot.num == 0:
                united_dataset = full_snapshot_dataset
            else:
                united_dataset = pd.concat([united_dataset, full_snapshot_dataset], axis=0)
            snapshot.num += 1

        united_dataset = united_dataset.reset_index()
        logging.debug(f'Final dataset: {united_dataset}')
        united_dataset.to_csv(self.output_path)
        logging.info(f'Saved dataset to {self.output_path}')


if __name__ == '__main__':
    setup_path = '../data_config.yaml'
    dataset_config_path = '../dataset_config.yaml'
    with open(setup_path) as stream:
        data_config = yaml.load(stream, yaml.Loader)

    with open(dataset_config_path) as stream:
        dataset_config = yaml.load(stream, yaml.Loader)
        # print(dataset_config['snapshot_features']["common"])

    new_dataset = SequoiaDataset(data_config, dataset_config)
    new_dataset.run()
