import os
import random

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
        self.max_snapshots_number = _data_config['basic']['max_snapshots_number']
        self.snapshot_initial_offset = _data_config['basic']['snapshot_initial_offset']
        self.snapshot_min_dur = _data_config['basic']['snapshot_min_duration']
        self.min_window = _data_config['basic']['min_window']
        self.random_snapshot = _data_config['basic']['random_snapshot']  # make random snapshot offset for each person

        self.data_load_date = _data_config['basic']['data_load_date']
        self.data_begin_date = self.str_to_datetime(_data_config['basic']['data_begin_date'])

        self.data_dir = self.data_config['data_location']['data_path']
        self.filename = self.data_config['data_location']['file_name']
        self.output_path = os.path.join(self.data_dir, self.data_config['data_location']['output_name'])

        # df2.merge(df1, how='union', on='ФИО')
        self.main_features = self.data_config['required_sheets']['basic']['features']
        self.common_features_name_mapping = {}
        self.continuous_features_names_mapping = {}
        self.specific_features = []

        # collect feature input names which are common for all snapshots:
        for f_name in self.main_features.keys():
            f = self.main_features[f_name]
            if 'kind' in f.keys():
                if f['kind'] == 'common':
                    self.common_features_name_mapping[f['name']] = f['name_out']
                elif f['kind'] == 'continuous':
                    self.continuous_features_names_mapping[f['name']] = f['name_out']
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
        new_columns = ['code']
        for year in [2022, 2023, 2024]:
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
        tot = 0
        for date, val in _values_per_month.items():  # same order of columns like in xlsx table
            if date <= _snapshot_start:
                tot += 1
                val = val.item()
                if val is None or val == np.NaN or str(val) in ['nan', 'NaN']:
                    continue
                val_item = str(val).split('\xa0')
                val_corr = val_item[0] + val_item[1] if len(val_item) > 1 else val_item[0]
                val_corr = float(val_corr.replace(',', '.'))
                # print("Val corr:", val_corr)
                values_sum += val_corr
                count += 1
                if tot == int(_period_months):
                    break
        if count == 0:
            return None
        avg = values_sum / count
        # print("avg:", avg)
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
        if self.random_snapshot:  # make random offset
            start_date = max(_recruitment_date, self.data_begin_date)
            empl_period = (_term_date - start_date).days / 30
            individual_max_initial_offset = int(empl_period - _snapshot.min_duration)
            _snapshot.initial_offset = random.randint(self.snapshot_initial_offset, individual_max_initial_offset)
            # print(f"init offset between {self.snapshot_initial_offset} and {individual_max_initial_offset} = {_snapshot.initial_offset}")

        offset_months = _snapshot.num * _snapshot.duration + _snapshot.initial_offset
        offset_years = int(np.floor(offset_months / 12))
        offset_months = offset_months % 12
        year = _term_date.year - offset_years
        month = _term_date.month - offset_months
        if month <= 0:
            month = 12 + month
            year -= 1
        # if year < _recruitment_date.year or (year == _recruitment_date.year and month < _recruitment_date.month + _snapshot.min_duration):
        #    return None
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
        return sample, snapshot_start, person_recruitment_date, person_termination_date

    def fill_average_values(self, _dataset: pd.DataFrame, _feature_df: pd.DataFrame, _longterm_avg: pd.DataFrame, _shortterm_avg: pd.DataFrame, _snapshot: SnapShot):
        count = 0

        for code in _dataset['code']:
            sample, snapshot_start, recr_date, dism_date = self.prepare_sample(_feature_df, _dataset, _snapshot, code)
            if not sample.empty and snapshot_start is not None:
                # print("Recr and dism dates:", recr_date, dism_date)
                # print("Snapshot start:", snapshot_start)
                longterm_period = min(_snapshot.duration, (snapshot_start - recr_date).days / 30)
                shortterm_period = min(self.min_window, (snapshot_start - recr_date).days / 30)
                # print("Longterm period:", longterm_period, 'shortterm:', shortterm_period)

                avg_6m = self.calc_numerical_average(sample, longterm_period, snapshot_start)
                avg_2m = self.calc_numerical_average(sample, shortterm_period, snapshot_start)
                _longterm_avg.loc[count] = [code, avg_6m]
                _shortterm_avg.loc[count] = [code, avg_2m]
            else:
                _longterm_avg.loc[count] = [code, None]
                _shortterm_avg.loc[count] = [code, None]


            count += 1
        return _shortterm_avg, _longterm_avg

    def calc_time_since_events(self, _events: IrregularEvents, _dataset: pd.DataFrame, _snapshot: SnapShot):
        time_since_event = pd.DataFrame({'code': [], _events.feature_name: []})
        count = 0

        for code in _dataset['code']:
            _, snapshot_start, person_recruitment_date, _ = self.prepare_sample(_events.events, _dataset, _snapshot, code)
            dates = _events.events_dates(code)

            time_since_salary_increase = self.calc_time_since_latest_event(dates, snapshot_start, person_recruitment_date)
            time_since_event.loc[count] = [code, time_since_salary_increase]

            count += 1

        return time_since_event

    def process_timeseries(self, _input_file: os.path, _dataset: pd.DataFrame, _snapshot: SnapShot, _sheet_name: str, _feature_name: str):
        df = read_excel(_input_file, sheet_name=_sheet_name)
        df = df.drop(columns='Full name')
        df = self.set_column_labels_as_dates(df)

        longterm_col = pd.DataFrame({'code': [], _feature_name + '_longterm': []})
        shortterm_col = pd.DataFrame({'code': [], _feature_name + '_shortterm': []})

        return self.fill_average_values(_dataset, df, longterm_col, shortterm_col, _snapshot)

    def calc_external_factors(self, _dataset: pd.DataFrame, _snapshot: SnapShot):
        inflation = read_excel('data/Инфляция.xlsx')
        unemployment = read_excel('data/Безработица.xlsx')
        bank_rate = read_excel('data/Ставка ЦБ.xlsx')

        factor_num = 0
        result = []
        for factor_data in [inflation, unemployment, bank_rate]:
            count = 0
            factor_num += 1
            factor_col = pd.DataFrame({'code': [], 'external_factor_'+str(factor_num): []})
            for code in _dataset['code']:
                _, snapshot_start, person_recruitment_date, _ = self.prepare_sample(_dataset, _dataset, _snapshot, code)
                values = factor_data.loc[factor_data['Год'] == snapshot_start.year].drop(columns='Год').values
                value = values[0][snapshot_start.month-1]
                factor_col.loc[count] = [code, value]
                count += 1
            result.append(factor_col)
        return result


    def process_continuous_features(self, _input_file: os.path, _dataset: pd.DataFrame, _snapshot: SnapShot, _sheet_name: str, _feature_name: str):
        data_file = pd.ExcelFile(_input_file)
        df_common = data_file.parse(sheet_name=_sheet_name)
        feature_col = pd.DataFrame({'code': [], self.continuous_features_names_mapping[_feature_name]: []})
        count = 0
        print('Continuous feature:', _feature_name)
        for code in _dataset['code']:
            _, snapshot_start, person_recruitment_date, person_termination_date = self.prepare_sample(_dataset, _dataset, _snapshot, code)
            value = df_common.loc[df_common['Code'] == code][_feature_name]
            value = value.replace(',', '.', regex=True).astype(float).item()
            value = abs(value)  # there are negative company seniority values in the data
            value -= (person_termination_date - snapshot_start).days / 365
            feature_col.loc[count] = [code, value]
            count += 1

        return [feature_col]


    def fill_snapshot_specific(self, _specific_features: list, _input_file: os.path, _dataset: pd.DataFrame, _full_dataset: pd.DataFrame, _snapshot: SnapShot):
        snapshot_columns = []

        for sheet_name, feature_name in self.time_series_name_mapping.items():
            print("process timeseries:", feature_name)
            snapshot_columns.extend(self.process_timeseries(_input_file, _dataset, _snapshot, sheet_name, feature_name))

        # age, overall experience, company seniority:
        for feature_name in ['Age', 'Seniority']:  # , 'overall_experience']:
            snapshot_columns.extend(
                self.process_continuous_features(_input_file, _dataset, _snapshot, 'Основные данные', feature_name))

        # ie = IrregularEvents(_input_file, 'days_since_promotion', 'Дата повышения', 'event_list')
        # snapshot_columns.extend([self.calc_time_since_events(ie, _dataset, _snapshot)])

        # ie = IrregularEvents(_input_file, 'time_since_salary_increase', 'Оплата труда', 'time_series')
        # snapshot_columns.extend([self.calc_time_since_events(ie, _dataset, _snapshot)])

        snapshot_columns.extend(self.calc_external_factors(_dataset, _snapshot))

        # sheet_name = 'Структура компании'
        # df = read_excel(_input_file, sheet_name=sheet_name)
        # df = df.drop(columns='№')
        # df.columns = ['code', 'department', 'manager']
        # count = 0
        #
        # for code in _dataset['code']:
        #     sample, snapshot_start, _, _ = self.prepare_sample(df, _dataset, _snapshot, code)
        #     manager_code = sample.iloc[0].values[-1]
        #     manager_sample = _full_dataset.loc[_full_dataset['code'] == manager_code]
        #     manager_term_date = manager_sample['termination_date'].item()
        #
        #     print("diff", snapshot_start - manager_term_date)
        #
        #     count += 1

        for new_col in snapshot_columns:
            _dataset = _dataset.merge(new_col, on='code', how='outer')

        # assing employee codes with current snapshot mark:
        _dataset = self.apply_snapshot_specific_codes(_dataset, _snapshot.num)

        return _dataset

    def fill_column(self, _f_name, _dataset, _col):
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
        russian = ['Россия', 'россия', 'Российское', 'российское', 'резидент', 'Резидент']
        not_russian = ['иное', 'НЕ РФ', 'НЕ Рф', 'Не резидент', 'не резидент']
        interm_education = ['среднее', 'Среднее', 'Без образования']
        high_education = ['высшее', 'Высшее']
        spec_education = ['среднее специальное', 'Среднее специальное']
        married = ['женат', 'замужем', 'в браке']
        single = ['не женат', 'не замужем', 'не в браке']
        office = ['office', 'accountant']
        sales = ['sales']
        blue_collar = ['blue collar']
        class_3_1 = ['Подкласс 3.1 класса условий труда "вредный"']
        class_2 = ['Допустимый, подкласс условий труда 2']

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
            elif key is None or str(key) == 'nan':
                return 3  # imputation
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
            if key in office:
                return 1
            elif key in sales:
                return 0
            elif key in blue_collar:
                return 2
            else:
                raise ValueError(f'Invalid department format: {key}')
        elif f_name == 'n_employers':
            #if not key.isdigit():
            #    raise TypeError(f'Unexpected n_employers value: {key}')
            return int(key)
        elif f_name == 'occupational_hazards':
            if key in class_3_1:
                return 0
            elif key in class_2:
                return 1
            elif key is None:
                return 3  # imputation
            else:
                raise ValueError(f'Invalid hazard format: {key}')
        elif f_name in ['recruitment_date', 'termination_date']:
            if key is None or str(key) == 'nan':  # no date of dismissal
                return None
            return self.str_to_datetime(str(key))
        elif f_name == 'status':
            return 1 - int(key)
        else:
            raise ValueError(f'Invalid feature name passed to lookup(): {f_name}')

    def fill_common_features(self, _features_names: dict, _input_df: pd.DataFrame, _target_dataset: pd.DataFrame):
        for n, col in _input_df.transpose().iterrows():  # transpose dataframe to iterate over columns
            if col.name in _features_names:
                new_col = col.values
                feature_name = _features_names[col.name]
                self.fill_column(feature_name, _target_dataset, new_col)
        return _target_dataset

    def collect_main_data(self, _common_features: {}, _input_df: pd.DataFrame, _data_config: dict, _snapshot: SnapShot):
        snapshot_dataset = pd.DataFrame()
        full_dataset = pd.DataFrame()

        full_dataset = self.fill_common_features(_common_features, _input_df, full_dataset)

        for n, row in _input_df.iterrows():
            recr_date = row['Hire date']
            term_date = row['Date of dismissal']
            if term_date is None or str(term_date) == 'nan':
                row['Date of dismissal'] = self.data_load_date
                term_date = row['Date of dismissal']
                _input_df.loc[n] = row
            empl_period = (self.str_to_datetime(term_date)-self.str_to_datetime(recr_date)).days / 30
            # remove sample if it has no data for current time snapshot:

            if empl_period - _snapshot.total_offset() <= _snapshot.min_duration:
                # print(empl_period, row['Seniority'])
                _input_df = _input_df.drop(axis='index', labels=n)
            elif (self.str_to_datetime(term_date) - self.data_begin_date).days / 30 - _snapshot.total_offset() <= _snapshot.min_duration:
                # print(term_date, row['Seniority'])
                _input_df = _input_df.drop(axis='index', labels=n)
            elif _snapshot.num > 0 and row['Status'] == 0:  # the employee is dismissed - don't include them into past snapshots
                _input_df = _input_df.drop(axis='index', labels=n)


        if _input_df.empty:
            snapshot_dataset = None
        else:
            snapshot_dataset = self.fill_common_features(_common_features, _input_df, snapshot_dataset)

        if _snapshot.num > 0:
            snapshot_dataset['status'] = [0 for i in range(len(snapshot_dataset['status']))]
            full_dataset['status'] = [0 for i in range(len(full_dataset['status']))]

        return snapshot_dataset, full_dataset

    def run(self):
        data_file_path = os.path.join(self.data_dir, self.filename)
        data_file = pd.ExcelFile(data_file_path)
        self.check_required_sheets(self.required_sheets, data_file.sheet_names)

        input_df_common = data_file.parse(sheet_name='Основные данные')

        snapshot = SnapShot(self.snapshot_duration, self.snapshot_min_dur, self.snapshot_initial_offset, 0)
        united_dataset = pd.DataFrame()
        while True:
            logging.info(f'Starting snapshot {snapshot.num}')
            main_dataset, main_dataset_full = self.collect_main_data(self.common_features_name_mapping, input_df_common, self.data_config, snapshot)

            if main_dataset is None:
                logging.info(f'No data left for snapshot {snapshot.num}. Finishing process...')
                break
            logging.debug(f'Formed main part of dataset: {main_dataset}')
            full_snapshot_dataset = self.fill_snapshot_specific(self.specific_features, data_file_path, main_dataset, main_dataset_full, snapshot)

            for n, row in full_snapshot_dataset.iterrows():
                if row.isnull().values.any():
                    print("NaN value in snapshot dataset")
                    full_snapshot_dataset = full_snapshot_dataset.drop(index=n)
            logging.debug(f'Final dataset for snapshot {snapshot.num}: {full_snapshot_dataset}')
            if snapshot.num == 0:
                united_dataset = full_snapshot_dataset
            else:
                united_dataset = pd.concat([united_dataset, full_snapshot_dataset], axis=0)
            snapshot.num += 1

            if snapshot.num == self.max_snapshots_number:
                break

        united_dataset = united_dataset.sort_values(by='code')  # sorting by code helps further splitting to train/val, keeping snapshots of each person in the same set to prevent data leakage
        united_dataset = united_dataset.drop(columns=['recruitment_date', 'termination_date', 'code'])
        cols = united_dataset.columns
        cols = cols.drop('status')
        cols = cols.tolist() + ['status']
        united_dataset = united_dataset[cols]
        # united_dataset = united_dataset.reset_index()
        logging.debug(f'\nFinal dataset:\n{united_dataset}')
        united_dataset.to_csv(self.output_path, index=False)
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
