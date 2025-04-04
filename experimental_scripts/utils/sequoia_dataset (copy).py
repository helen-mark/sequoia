import os

from datetime import datetime, date
import pandas as pd
import numpy as np
import yaml
from pandas import read_excel

import logging
logging.basicConfig(level=logging.DEBUG)


""" Class Snapshot stores information about snapshot of working time.
 Each snapshot has an offset of n months to the past from the work termination date of a worker who left,
 or the data dumping day for a present worker.
 When we jump N months to the past, we "forget" those months of work and consider all the relevant information
 for the worker on the start day of the snapshot.
 So we can take several snapshots for each worker as separate data samples as a way of data augmentation and
 a way of avoiding overfitting.
 """
class SnapShot:
    def __init__(self, _dur: int, _min_dur: int, _initial_offset: int, _snapshot_num: int):
        self.duration = _dur  # amount of days
        self.min_duration = _min_dur
        self.initial_offset = _initial_offset  # amount of days
        self.num = _snapshot_num

    def total_offset(self):
        return self.initial_offset + self.duration*self.num


""" Irregular events like salary increase or promotions.
The initial data can be either represented a table of salary-per-month for each worker to figure out 
salary increase days from it ("time series"), or a table containing promotion dates as a list for each worker
("event list")
"""
class IrregularEvents:
    def __init__(self, _input_file: str, _feature_name: str, _sheet_name: str, _kind: str):
        self.feature_name = _feature_name
        self.sheet_name = _sheet_name
        self.kind = _kind
        df = read_excel(_input_file, sheet_name=_sheet_name)
        df = df.drop(columns='№')
        if _kind == 'event_list':
            df.columns = ['code', _sheet_name]
        elif _kind == 'time_series':
            df = SequoiaDataset.set_column_labels_as_dates(df)
        else:
            raise ValueError(f'Unexpected event kind: {_kind}')
        self.events = df

    def events_dates(self, _code: int):
        sample = self.events.loc[self.events['code'] == _code]
        sample = sample.drop(columns='code')  # leave ony columns with salary values

        if self.kind == 'time_series':
            dates = self.calc_value_increase_dates(sample)
        elif self.kind == 'event_list':
            dates_str = sample.iloc[0].values
            dates = [SequoiaDataset.str_to_datetime(x) for x in dates_str]
        return dates

    def calc_value_increase_dates(self, _values_per_month: pd.DataFrame):
        dates = []
        assert not _values_per_month.empty

        salary_prev = 0
        for name, val in _values_per_month.items():  # same order of columns like in xlsx table
            if val.item() > salary_prev:  # consider employers' first salary as increase too
                salary_prev = val.item()
                dates.append(name)
        return dates


""" Class SequoiaDataset is responsible for creating a dataset from raw data dump got from client.
Requirements to the raw data representation can be found in config files
"""
class SequoiaDataset:
    def __init__(self, _data_config, _dataset_config):
        logging.info("Initializing new dataset instance...")
        self.data_config = _data_config
        self.dataset_config = _dataset_config
        self.snapshot_duration = _data_config['basic']['snapshot_duration']
        self.snapshot_initial_offset = _data_config['basic']['snapshot_initial_offset']
        self.snapshot_min_dur = _data_config['basic']['snapshot_min_duration']

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
        self.events_name_mapping = {}
        self.required_sheets = []
        for key in self.data_config['required_sheets']:
            self.required_sheets.append(self.data_config['required_sheets'][key]['name'])
            if 'kind' in self.data_config['required_sheets'][key].keys():
                if 'time_series' in self.data_config['required_sheets'][key]['kind']:
                    self.time_series_name_mapping[self.data_config['required_sheets'][key]['name']] = self.data_config['required_sheets'][key]['name_out']
                if 'events' in self.data_config['required_sheets'][key]['kind']:
                    self.events_name_mapping[self.data_config['required_sheets'][key]['name']] = \
                    self.data_config['required_sheets'][key]['name_out']

    def check_required_sheets(self, _required_sheets: [], _actual_sheets: []):
        absent = []
        for s in _required_sheets:
            if s not in _actual_sheets:
                absent.append(s)
        if len(absent) > 0:
            raise Exception(f'Required sheets absent in input file: {absent}\nPresent sheetnames: {_actual_sheets}')

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

    @staticmethod
    def set_column_labels_as_dates(_input_df: pd.DataFrame):
        year = 2024
        new_columns = ['code']
        for i in range(1, 13):
            new_columns.append(date(year, i, 1))
        _input_df.columns = new_columns
        return _input_df


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

    def calc_time_since_latest_event(self, _event_dates: list, _snapshot_start: datetime.date, _recruitment_date: datetime.date):
        _event_dates.reverse()
        for date in _event_dates:
            if date < _snapshot_start:
                return _snapshot_start - date
        return _snapshot_start - _recruitment_date

    def check_leader_left(self):
        pass

    def check_has_meal(self):
        pass

    def check_has_insurance(self):
        pass

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

    def apply_snapshot_specific_codes(self, _dataset: pd.DataFrame, _snapshot_num: int):
        codes = _dataset['code']
        new_codes = []
        for code in codes:
            new_codes.append(str(code) + '_s' + str(_snapshot_num))
        _dataset['code'] = new_codes
        return _dataset

    @staticmethod
    def str_to_datetime(_date_str: str):
        splt = _date_str.split('.')
        return date(int(splt[2]), int(splt[1]), int(splt[0]))

    def prepare_sample(self, _feature_df: pd.DataFrame, _dataset: pd.DataFrame, _snapshot: SnapShot, _code: int):
        sample = _feature_df.loc[_feature_df['code'] == _code]
        sample = sample.drop(columns='code')  # leave ony columns with salary values
        person_recruitment_date = _dataset.loc[_dataset['code'] == _code]['recruitment_date'].item()
        person_termination_date = _dataset.loc[_dataset['code'] == _code]['termination_date'].item()

        snapshot_start = self.calc_individual_snapshot_start(_snapshot, person_recruitment_date,
                                                             person_termination_date)
        return sample, snapshot_start, person_recruitment_date

    def fill_average_values(self, _dataset: pd.DataFrame, _feature_df: pd.DataFrame, _longterm_avg: pd.DataFrame, _shortterm_avg: pd.DataFrame, _snapshot: SnapShot):
        count = 0

        for code in _dataset['code']:
            sample, snapshot_start, _ = self.prepare_sample(_feature_df, _dataset, _snapshot, code)

            avg_6m = self.calc_numerical_average(sample, 6, snapshot_start)
            avg_2m = self.calc_numerical_average(sample, 2, snapshot_start)
            _longterm_avg.loc[count] = [code, avg_6m.item()]
            _shortterm_avg.loc[count] = [code, avg_2m.item()]

            count += 1

        return _shortterm_avg, _longterm_avg

    def calc_time_since_events(self, _events: IrregularEvents, _dataset: pd.DataFrame, _snapshot: SnapShot):
        time_since_event = pd.DataFrame({'code': [], _events.feature_name: []})
        count = 0

        for code in _dataset['code']:
            _, snapshot_start, person_recruitment_date = self.prepare_sample(_events.events, _dataset, _snapshot, code)
            dates = _events.events_dates(code)

            time_since_salary_increase = self.calc_time_since_latest_event(dates, snapshot_start, person_recruitment_date)
            time_since_event.loc[count] = [code, time_since_salary_increase]

            count += 1

        return time_since_event

    def process_timeseries(self, _input_file: os.path, _dataset: pd.DataFrame, _snapshot: SnapShot, _sheet_name: str, _feature_name: str):
        df = read_excel(_input_file, sheet_name=_sheet_name)

        df = df.drop(columns='№')
        df = self.set_column_labels_as_dates(df)

        longterm_col = pd.DataFrame({'code': [], _feature_name + '_longterm': []})
        shortterm_col = pd.DataFrame({'code': [], _feature_name + '_shortterm': []})

        return self.fill_average_values(_dataset, df, longterm_col, shortterm_col, _snapshot)

    def fill_snapshot_specific(self, _specific_features: list, _input_file: os.path, _dataset: pd.DataFrame, _snapshot: SnapShot):
        snapshot_columns = []

        for sheet_name, feature_name in self.time_series_name_mapping.items():
            snapshot_columns.extend(self.process_timeseries(_input_file, _dataset, _snapshot, sheet_name, feature_name))

        ie = IrregularEvents(_input_file, 'days_since_promotion', 'Дата повышения', 'event_list')
        snapshot_columns.extend([self.calc_time_since_events(ie, _dataset, _snapshot)])

        ie = IrregularEvents(_input_file, 'time_since_salary_increase', 'Оплата труда', 'time_series')
        snapshot_columns.extend([self.calc_time_since_events(ie, _dataset, _snapshot)])

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
        data_file = pd.ExcelFile(data_file_path)
        self.check_required_sheets(self.required_sheets, data_file.sheet_names)

        input_df_common = data_file.parse(sheet_name='Основные данные')

        snapshot_num = 0
        snapshot = SnapShot(self.snapshot_duration, self.snapshot_min_dur, self.snapshot_initial_offset, snapshot_num)
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
