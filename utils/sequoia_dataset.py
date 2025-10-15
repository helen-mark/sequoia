import os
import random

from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np
import yaml
from pandas import read_excel
from dateutil.relativedelta import relativedelta  # Handles month subtraction correctly

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
        self.step = _dur  # amount of days
        self.min_duration = _min_dur
        self.initial_offset = _initial_offset  # amount of days
        self.num = _snapshot_num

    def total_offset(self):
        return self.initial_offset + self.step*self.num


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

    def get(self):
        return self.recr_date, self.term_date, self.snapshot_start



""" Class SequoiaDataset is responsible for creating a dataset from raw data dump got from client.
Requirements to the raw data representation can be found in config files
"""
class SequoiaDataset:

    def __init__(self, _data_config, _dataset_config):
        logging.info("Initializing new dataset instance...")
        self.data_config = _data_config
        self.dataset_config = _dataset_config
        self.snapshot_step = _data_config['time_snapshots']['snapshot_step']
        self.max_snapshots_number = _data_config['time_snapshots']['max_snapshots_number']
        self.snapshot_initial_offset = _data_config['time_snapshots']['snapshot_initial_offset']
        self.snapshot_min_dur = _data_config['time_snapshots']['snapshot_min_duration']
        self.min_window = _data_config['time_snapshots']['min_window']
        self.max_window = _data_config['time_snapshots']['max_window']
        self.random_snapshot = _data_config['time_snapshots']['random_snapshot']  # make random snapshot offset for each person

        self.forecast_horison = _data_config['options']['forecast_horison']
        self.remove_censored = _data_config['options']['remove_censored']

        self.data_load_date = pd.to_datetime(_data_config['data']['data_load_date'], dayfirst=True).date()
        self.data_begin_date = pd.to_datetime(_data_config['data']['data_begin_date'], dayfirst=True).date()
        self.remove_short_service = _data_config['data']['remove_short_service']

        self.industry = _data_config['data']['industry']
        self.region = _data_config['data']['region']
        self.calibrate_by = _data_config['data']['income_calibration']

        self.data_dir = self.data_config['data_location']['data_path']
        self.filename = self.data_config['data_location']['file_name']
        self.output_path = os.path.join(self.data_config['data_location']['trg_path'], self.data_config['data_location']['output_name'])

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

    def get_industry_label(self):
        if self.industry == 'Metals':
            return 0
        elif self.industry == 'Light industry':
            return 1
        elif self.industry == 'Electric':
            return 2
        elif self.industry == 'Chemistry':
            return 3
        else:
            raise ValueError(f'Invalid industry: {self.industry}')

    def get_region_label(self):
        if self.region == 'Moscow':
            return 0
        elif self.region == 'Orenburg':
            return 1
        else:
            raise ValueError(f'Invalid region: {self.region}')

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
                new_columns.append(date(year=year, month=i, day=1))
        _input_df.columns = new_columns
        return _input_df

    # def calc_derivative(self, _values_per_month: pd.DataFrame, _period_months: int, _timepoint_in_past: datetime.date, _cur_val: float):
    #     prev_val = self.calc_numerical_average(_values_per_month, _period_months, _timepoint_in_past)
    #     if prev_val is None:
    #         return None
    #     return _cur_val - prev_val


    def calc_numerical_average(self, _values_per_month: pd.DataFrame, _period_months: int, _snapshot_timepoint: datetime.date, _recr_date: datetime.date, _term_date: datetime.date):
        assert not _values_per_month.empty
        # assert _period_months > 0
        values_reversed = _values_per_month[_values_per_month.columns[::-1]]  # this operation reverses order of columns
        values_sum = 0.
        count = 0
        tot = 0
        values = [x for x in values_reversed.values[0] if str(x).lower() != 'nan']

        def update_sum(val, sum):
            val = val.item()
            if val is None or val == np.nan or str(val).lower() == 'nan':
                return None

            val_item = str(val).split('\xa0')
            val_corr = val_item[0] + val_item[1] if len(val_item) > 1 else val_item[0]
            val_corr = float(val_corr.replace(',', '.'))
            # print("Val corr:", val_corr)
            sum += val_corr
            return sum

        # avg_val = np.ma.average(values)
        for date, val in values_reversed.items():  # same order of columns like in xlsx table
            if date <= _snapshot_timepoint and date > _recr_date and date < _term_date.replace(day=1):  # don't take first and last months of the employee! the salary is incomplete
                tot += 1
                if tot > int(_period_months):
                    break
                sum = update_sum(val, values_sum)
                if sum is None:
                    continue
                values_sum = sum
                count += 1


        if count == 0:
            logging.info("Found 0 valid values in time series. Performing imputation...")
            for date, val in values_reversed.items():  # same order of columns like in xlsx table
                if date > _recr_date and date < _term_date.replace(day=1):  # don't take first and last months of the employee! the salary is incomplete
                    sum = update_sum(val, values_sum)
                    if sum is None:
                        continue
                    values_sum = sum
                    count += 1
                    if count >= int(_period_months):
                        break

        if count == 0:
            logging.info("Found 0 valid values in time series. Failed to impute")
            return None
        avg = values_sum / count
        # logging.info(f"Successful imputation with value {avg}")
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

    def age_from_birth_date(self, _dataset: pd.DataFrame):
        default_date = date(year=1975, month=1, day=1)
        _dataset['birth_date'] = _dataset['birth_date'].fillna(default_date)
        bd_col = _dataset['birth_date']
        cur_date_col = _dataset['termination_date']
        age_col = (pd.to_datetime(cur_date_col) - pd.to_datetime(bd_col)).dt.days / 365.
        _dataset.insert(len(_dataset.columns), 'age', age_col)
        return _dataset


    def fill_status(self, _dataset: pd.DataFrame):
        cur_date_col = _dataset['termination_date']
        statuses = []
        for i, d in enumerate(cur_date_col.values):
            if pd.isnull(d) or d == self.data_load_date:  # no termination date - person still working
                statuses.append(0)
            else:
                statuses.append(1)
        _dataset = _dataset.drop(columns=['status'], errors='ignore')
        _dataset.insert(len(_dataset.columns), 'status', np.array(statuses))
        return _dataset


    def seniority_from_term_date(self, _dataset: pd.DataFrame):
        cur_date_col = _dataset['termination_date']
        recr_date_col = _dataset['recruitment_date']
        seniority_col = (cur_date_col - recr_date_col).apply(lambda x: x.days) / 365.
        _dataset = _dataset.drop(columns=['seniority'], errors='ignore')
        _dataset.insert(len(_dataset.columns), 'seniority', seniority_col)
        for v in _dataset['seniority'].values:
            if v > 30:
                print("Suspicious seniority value:", v)
        return _dataset

    def apply_snapshot_specific_codes(self, _dataset: pd.DataFrame, _snapshot_num: int):
        logging.info('Applying snapshot specific codes...')
        codes = _dataset['code']
        new_codes = []
        for code in codes:
            new_codes.append(str(code) + '_s' + str(_snapshot_num))
        _dataset['code'] = new_codes
        return _dataset

    @staticmethod
    def str_to_datetime(_date_str: str):
        splt = _date_str.split('.')
        if len(splt) < 3:
            splt = _date_str.split('/')
            if len(splt) < 3:
                splt = _date_str.split('-')
                return date(int(splt[0]), int(splt[1]), int(splt[2][:2]))
            return date(int(splt[0]), int(splt[1]), int(splt[2]))  # european order
        return date(int(splt[2]), int(splt[1]), int(splt[0][:2]))  # russian order

    def prepare_sample(self, _feature_df: pd.DataFrame, _dataset: pd.DataFrame, _snapshot: SnapShot, _code: int):
        sample = _feature_df.loc[_feature_df['code'] == _code]
        if sample.empty:
            print(f"Empty sample in function prepare_sample for {_snapshot.num} snapshot: {_code}")
            # logging.info(f"Code {_code} is not present in feature dataframe! Empty sample")
        sample = sample.drop(columns='code')  # leave ony columns with salary values
        # person_recruitment_date = _dataset.loc[_dataset['code'] == _code]['recruitment_date'].item()
        # person_termination_date = _dataset.loc[_dataset['code'] == _code]['termination_date'].item()
        #
        # snapshot_start = self.calc_individual_snapshot_start(_snapshot, person_recruitment_date,
        #                                                      person_termination_date)
        return sample  # , snapshot_start, person_recruitment_date, person_termination_date

    def extract_features_from_timeseries(self, _dataset: pd.DataFrame, _feature_df: pd.DataFrame, _longterm_avg: pd.DataFrame, _shortterm_avg: pd.DataFrame, _deriv_col: pd.DataFrame, _snapshot: SnapShot):
        count = 0

        for code in _dataset['code']:
            sample = self.prepare_sample(_feature_df, _dataset, _snapshot, code)
            snapshot_timepoint = _dataset[_dataset['code']==code]['snapshot_start'].item()
            start_date = _dataset[_dataset['code']==code]['start_date'].item()
            end_date = _dataset[_dataset['code']==code]['end_date'].item()
            recr_date = _dataset[_dataset['code']==code]['recruitment_date'].item()
            term_date = _dataset[_dataset['code']==code]['termination_date'].item()

            if (_dataset[_dataset['code'] == code]['status'].item()):
                if snapshot_timepoint.month >= _dataset[_dataset['code']==code]['termination_date'].item().month:
                    logging.info("Critical! No snapshot offset from term date")
                    _longterm_avg.loc[count] = [code, None]
                    _shortterm_avg.loc[count] = [code, None]
                    _deriv_col.loc[count] = [code, None]

            if not sample.empty and snapshot_timepoint is not None:
                # longterm_period = min(self.max_window, (snapshot_timepoint - start_date).days / 30)
                shortterm_period = min(self.min_window, (snapshot_timepoint - start_date).days / 30.4)
                avg_now = self.calc_numerical_average(sample, shortterm_period, snapshot_timepoint, recr_date, term_date)

                past_timepoint = snapshot_timepoint - relativedelta(months=3)  # 3 months is how long to the past we want to move
                past_timepoint = max(past_timepoint, start_date + timedelta(days=_snapshot.min_duration * 30.4))
                dt = (snapshot_timepoint - past_timepoint).days / 365.  # Time delta in years
                avg_past = self.calc_numerical_average(sample, shortterm_period, past_timepoint, recr_date, term_date)
                # if avg_now == 0:
                #     print('status', _dataset[_dataset['code'] == code]['status'].item())
                if avg_now is None or avg_past is None:
                    print(f"Numerical average for {code} is None")
                    deriv = None
                elif dt > 0 or dt >= 1. / 4.:  # we have at least 3 months to calculate derivative
                    deriv = (avg_now - avg_past)
                else:
                   # print('--------------------------- dt <=0 ! ', past_timepoint, start_date, preset.get())
                    deriv = -1000000  # consider time period too small


                _longterm_avg.loc[count] = [code, avg_past]
                _shortterm_avg.loc[count] = [code, avg_now]
                _deriv_col.loc[count] = [code, deriv]
            else:
                _longterm_avg.loc[count] = [code, None]
                _shortterm_avg.loc[count] = [code, None]
                _deriv_col.loc[count] = [code, None]

            count += 1
        # deriv_col = pd.DataFrame({'code': _longterm_avg['code'],
        #                           _feature_name + '_deriv': (_shortterm_avg.values[:, 1] - _longterm_avg.values[:,
        #                                                                                  1] / dt)})
        return _shortterm_avg, _longterm_avg, _deriv_col

    def calc_time_since_events(self, _events: IrregularEvents, _dataset: pd.DataFrame, _snapshot: SnapShot):
        time_since_event = pd.DataFrame({'code': [], _events.feature_name: []})
        count = 0

        for code in _dataset['code']:
            start_date = _dataset[_dataset['code']==code]['start_date']

            snapshot_timepoint = _dataset[_dataset['code']==code]['snapshot_start']
            dates = _events.events_dates(code)

            time_since_salary_increase = self.calc_time_since_latest_event(dates, snapshot_timepoint, start_date)
            time_since_event.loc[count] = [code, time_since_salary_increase]

            count += 1

        return time_since_event

    def process_timeseries(self, _df: pd.DataFrame, _dataset: pd.DataFrame, _snapshot: SnapShot, _sheet_name: str, _feature_name: str):
        # df = df.drop(columns='Full name')
        df = self.set_column_labels_as_dates(_df)

        longterm_col = pd.DataFrame({'code': [], _feature_name + '_longterm': []})
        shortterm_col = pd.DataFrame({'code': [], _feature_name + '_shortterm': []})
        deriv_col = pd.DataFrame({'code': [], _feature_name + '_deriv': []})

        shortterm_col, longterm_col, deriv_col = self.extract_features_from_timeseries(_dataset, df, longterm_col, shortterm_col, deriv_col, _snapshot)
        return shortterm_col, longterm_col, deriv_col

    def calc_external_factors(self, _dataset: pd.DataFrame, _snapshot: SnapShot):
        logging.info('Processing external factors...')
        inflation = read_excel('data_raw/Инфляция.xlsx')
        unemployment = read_excel('data_raw/Безработица.xlsx')
        bank_rate = read_excel('data_raw/Ставка ЦБ.xlsx')

        factor_num = 0
        result = []
        for factor_data in [inflation, unemployment, bank_rate]:
            count = 0
            factor_num += 1
            factor_col = pd.DataFrame({'code': [], 'external_factor_'+str(factor_num): []})
            for code in _dataset['code']:
                snapshot_timepoint = _dataset[_dataset['code']==code]['snapshot_start'].item()
                values = factor_data.loc[factor_data['Год'] == snapshot_timepoint.year].drop(columns='Год').values
                value = values[0][snapshot_timepoint.month-1]
                factor_col.loc[count] = [code, value]
                count += 1
            result.append(factor_col)
        return result


    def process_continuous_features(self, _dataset: pd.DataFrame, _snapshot: SnapShot, _feature_name: str):
        # feature_col = pd.DataFrame({'code': [], self.continuous_features_names_mapping[_feature_name]: []})
        count = 0
        logging.info(f'Processing continuous feature: {_feature_name}')
        for n, row in _dataset.iterrows():
            code = row['code']
            person_termination_date = _dataset[_dataset['code']==code]['termination_date'].item()
            snapshot_timepoint = _dataset[_dataset['code']==code]['snapshot_start'].item()
            value = _dataset.loc[_dataset['code'] == code][_feature_name]
            value = value.replace(',', '.', regex=True).astype(float).item()
            value = abs(value)  # there are negative company seniority values in the data
            value -= (person_termination_date - snapshot_timepoint).days / 365
            # feature_col.loc[count] = [code, value]
            row[_feature_name] = value
            count += 1
            _dataset.loc[n] = row

        return


    def fill_snapshot_specific(self, _specific_features: list, _input_file: os.path, _dataset: pd.DataFrame, _snapshot: SnapShot):
        snapshot_columns = []

        for sheet_name, feature_name in self.time_series_name_mapping.items():
            logging.info(f"Process timeseries: {feature_name}")
            df = read_excel(_input_file, sheet_name=sheet_name)
            snapshot_columns.extend(self.process_timeseries(df, _dataset, _snapshot, sheet_name, feature_name))


        # age, overall experience, company seniority:
        for feature_name in ['age', 'seniority', 'total_seniority']:  # , 'overall_experience']:
            if feature_name in _dataset.columns:
                self.process_continuous_features(_dataset, _snapshot, feature_name)

        # ie = IrregularEvents(_input_file, 'days_since_promotion', 'Дата повышения', 'event_list')
        # snapshot_columns.extend([self.calc_time_since_events(ie, _dataset, _snapshot)])

        # ie = IrregularEvents(_input_file, 'time_since_salary_increase', 'Оплата труда', 'time_series')
        # snapshot_columns.extend([self.calc_time_since_events(ie, _dataset, _snapshot)])

        snapshot_columns.extend(self.calc_external_factors(_dataset, _snapshot))

        for new_col in snapshot_columns:
            _dataset = _dataset.merge(new_col, on='code', how='outer')

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
        russian = ['Россия', 'россия', 'Российское', 'российское', 'РОССИЯ', 'резидент', 'Резидент']
        not_russian = ['иное', 'НЕ РФ', 'НЕ Рф', 'Не резидент', 'не резидент', 'УКРАИНА', 'КАЗАХСТАН', 'ТАДЖИКИСТАН', 'УЗБЕКИСТАН', 'МОЛДОВА, РЕСПУБЛИКА', 'БЕЛАРУСЬ', 'Украина', 'КИРГИЗИЯ', 'КЫРГЫЗСТАН']
        interm_education = ['среднее', 'Среднее (полное) общее образование', 'Среднее', 'Без образования', 'Среднее общее образование', 'Основное общее образование']
        high_education = ['высшее', 'Неполное высшее образование', 'Высшее', 'Высшее-бакалавриат', 'Высшее образование - бакалавриат', 'Высшее профессиональ', 'Высшее-магистратура', 'Высшее образование - специалитет, магистратура', 'Высшее-специалитет', 'Послевузов. професс']
        spec_education = ['среднее специальное', 'Профессиональное обучение', 'Начальное профессиональное образование', 'Среднее специальное', 'Среднее профессиональное образование', 'Среднее профессионал', 'Дополнительное профессиональное образование']
        married = ['женат', 'замужем', 'в браке', 'Состоит в зарегистрированном браке', 'Жен/Зм']
        single = ['не женат', 'не замужем', 'не в браке', 'Разв.', 'Х/НЗ']
        office = ['office', 'accountant']
        sales = ['sales']
        blue_collar = ['blue collar']
        class_3_1 = ['Подкласс 3.1 класса условий труда "вредный"', '3.1 Подкласс']
        class_3_2 = ['3.2 Подкласс']
        class_3_3 = ['3.3 Подкласс']
        class_2 = ['Допустимый, подкласс условий труда 2', '23200000-11618', '23200000-19756', '2 Класс']
        class_4 = ['Опасный, подкласс условий труда 4']

        if f_name == 'code':
            return key
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
                return 1
                # raise ValueError(f'Invalid citizenship format: {key}')
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
                return 2
                #raise ValueError(f'Invalid family status format: {key}')
        elif f_name == 'children':
            #if not key.isdigit():
            #    raise TypeError(f'Unexpected n_children value: {key}')
            return int(key)
        elif f_name == 'to_work_travel_time':
            return float(key)
        elif f_name == 'n_employers':
            #if not key.isdigit():
            #    raise TypeError(f'Unexpected n_employers value: {key}')
            return int(key)
        elif f_name == 'occupational_hazards':
            if key in class_3_1:
                return 3
            elif key in class_3_2:
                return 32
            elif key in class_3_3:
                return 33
            elif key in class_2:
                return 2
            elif key in class_4:
                return 4
            elif pd.isnull(key):
                return 0  # imputation
            else:
                raise ValueError(f'Invalid hazard format: {key}')
        elif f_name in ['recruitment_date', 'termination_date', 'birth_date']:
            if key is None or str(key) == 'nan' or pd.isna(key):  # no date of dismissal
                return None
            return self.str_to_datetime(str(key))
        elif f_name == 'status':
            return 1 - int(key)
        elif f_name in ['age', 'seniority', 'city', 'region', 'job_category', 'department', 'industry_kind_internal']:
            return key
        elif f_name == 'city_population':
            return int(key)
        elif f_name == 'total_seniority':
            return int(key.split('г')[0])
        else:
            raise ValueError(f'Invalid feature name passed to lookup(): {f_name}')

    def fill_common_features(self, _features_names: dict, _input_df: pd.DataFrame, _target_dataset: pd.DataFrame):
        for n, col in _input_df.transpose().iterrows():  # transpose dataframe to iterate over columns
            if col.name in _features_names:
                new_col = col.values
                feature_name = _features_names[col.name]
                self.fill_column(feature_name, _target_dataset, new_col)
        return _target_dataset

#    def calc_snapshot_start_vectorized(
#             self,
#             _df: pd.DataFrame,
#             _snapshot: SnapShot
#     ) -> pd.Series:
#         """
#         Vectorized replacement for calc_individual_snapshot_start().
#         Returns a Pandas Series of `date` objects.
#         """
#         # Convert dates to numpy.datetime64 for vectorized arithmetic
#         #start_date_np = pd.to_datetime(df['start_date']).values
#         #end_date_np = pd.to_datetime(df['end_date']).values
#
#         # Compute employment period in months
#         _df['empl_period'] = (_df['end_date'] - _df['start_date']) / 30.4
#
#         # Initialize snapshot_start with end_date (default for short periods)
#         _df['snapshot_start'] = _df['end_date']
#
#         # Mask for employees with sufficient work history
#         _df['long_enough_service'] = _df['empl_period'] - _snapshot.total_offset() >= _snapshot.min_duration
#
#         if self.random_snapshot:
#             # Compute random initial_offset where applicable
#             _df['individual_max_offset'] = np.floor(_df.loc[_df['long_enough_service']]['empl_period'] - _snapshot.min_duration).astype(int)
#             _df['offset_months'] = np.where(
#                 _df['individual_max_offset']>0,
#                 np.random.randint(_snapshot.initial_offset, _df['individual_max_offset'] + 1),
#                 _snapshot.initial_offset
#             )
#         else:
#             _df['offset_months'] = _snapshot.num * _snapshot.step + _snapshot.initial_offset

    def calc_snapshot_start_vectorized(
            self,
            df: pd.DataFrame,
            snapshot: SnapShot
    ) -> pd.Series:
        """
        Vectorized replacement for calc_individual_snapshot_start().
        Returns a Pandas Series of `date` objects.
        """
        # Convert dates to numpy.datetime64 for vectorized arithmetic
        start_date_np = pd.to_datetime(df['start_date']).values
        end_date_np = pd.to_datetime(df['end_date']).values

        # Compute employment period in months
        empl_period = (end_date_np - start_date_np) / np.timedelta64(30, 'D')

        # Initialize snapshot_start with end_date minus 1 month (default for short periods)
        one_month = np.timedelta64(1, 'M')
        snapshot_start = end_date_np.astype('datetime64[M]') - one_month
        snapshot_start = snapshot_start.astype('datetime64[D]')

        # Mask for employees with sufficient work history
        mask_long_enough = empl_period - snapshot.total_offset() >= snapshot.min_duration

        if self.random_snapshot:
            # Compute random initial_offset where applicable
            individual_max_offset = np.floor(empl_period[mask_long_enough] - snapshot.min_duration).astype(int)
            valid_mask = individual_max_offset > 0
            total_offset = np.where(
                valid_mask,
                np.random.randint(snapshot.initial_offset, individual_max_offset + 1),
                snapshot.initial_offset
            )
            offset_months = total_offset  # num = 0 for random snapshots
        else:
            offset_months = snapshot.num * snapshot.step + snapshot.initial_offset

        # Apply month/year arithmetic only to employees with sufficient history
        if np.any(mask_long_enough):
            # Convert total_offset_months to a NumPy array first
            total_offset_months_arr = np.array(offset_months, dtype='int64')
            month_offset = (-total_offset_months_arr).astype('timedelta64[M]')

            # Apply offset
            snapshot_start[mask_long_enough] = (
                    end_date_np[mask_long_enough].astype('datetime64[M]')
                    + month_offset
                    + np.timedelta64(1, 'D')  # Set day=1
            ).astype('datetime64[D]')

        # Handle edge cases where snapshot_start <= start_date
        invalid_mask = snapshot_start <= start_date_np
        if np.any(invalid_mask):
            logging.critical(
                f"Snapshot start coincides with data beginning date for {np.sum(invalid_mask)} employees. "
                "Defaulting to end_date minus 1 month"
            )
            # Already defaulted to end_date minus 1 month, no need to change

        # Ensure snapshot_start is always at least 1 month before end_date
        too_late_mask = snapshot_start >= (end_date_np.astype('datetime64[M]') - one_month).astype('datetime64[D]')
        if np.any(too_late_mask):
            logging.warning(
                f"Adjusting {np.sum(too_late_mask)} snapshot starts to be at least 1 month before end_date"
            )
            # Set to end_date minus 1 month
            snapshot_start[too_late_mask] = (
                    end_date_np[too_late_mask].astype('datetime64[M]') - one_month
            ).astype('datetime64[D]')

        # Convert to date objects
        def datetime64_to_date(dt64):
            return dt64.astype('datetime64[D]').item()  # Convert to date

        # Ensure the array is datetime64 (handles mixed input)
        for i, s in enumerate(snapshot_start):
            snapshot_start[i] = s.astype('datetime64[D]').item()
        # Apply conversion
        return pd.Series(snapshot_start, index=df.index)

    def collect_main_data(self, _common_features: {}, _input_df: pd.DataFrame, _data_config: dict, _snapshot: SnapShot):
        snapshot_dataset = pd.DataFrame()
        full_dataset = pd.DataFrame()

        full_dataset = self.fill_common_features(_common_features, _input_df, full_dataset)
        deleted_count = 0

        _input_df['Hire date'] = pd.to_datetime(_input_df['Hire date']).dt.date
        _input_df['Date of dismissal'] = pd.to_datetime(_input_df['Date of dismissal']).dt.date
        self.data_load_date = pd.to_datetime(self.data_load_date).date()  # Convert to Python date

        for n, row in _input_df.iterrows():
            recr_date = row['Hire date']
            term_date = row['Date of dismissal']
            code = row["Code"]
            status = 1

            if term_date is None or str(term_date) == 'nan' or pd.isna(term_date):
                row['Date of dismissal'] = self.data_load_date
                term_date = self.data_load_date
                _input_df.loc[n] = row
                status = 0
            if pd.isna(recr_date):
                _input_df = _input_df.drop(axis='index', labels=n)
                continue

            #recr_date = self.str_to_datetime(str(recr_date))
            #term_date = self.str_to_datetime(str(term_date))
            empl_period = (term_date-recr_date).days / 30.4
            start_date = max(recr_date, self.data_begin_date)
            end_date = min(term_date, self.data_load_date)

            # remove sample if it has no data for current time snapshot:

            if self.remove_short_service:
                if empl_period < 3:
                    _input_df = _input_df.drop(axis='index', labels=n)
                    deleted_count += 1
                    continue
                if empl_period - _snapshot.total_offset() <= _snapshot.min_duration:
                    # print(code)  # , recr_date, term_date, f"({int(empl_period)} months)")
                    _input_df = _input_df.drop(axis='index', labels=n)
                    deleted_count += 1
                    continue
                elif (term_date - self.data_begin_date).days / 30 - _snapshot.num * _snapshot.step <= _snapshot.min_duration:
                    _input_df = _input_df.drop(axis='index', labels=n)
                    deleted_count += 1
                    continue

            # cut out employees with employment beyond chosen data period:
            if recr_date >= self.data_load_date or term_date < self.data_begin_date:
                _input_df = _input_df.drop(axis='index', labels=n)
                deleted_count += 1
            elif _snapshot.num > 0 and (end_date - start_date).days / 30 <= _snapshot.total_offset():  # employment doesn't belong to this snapshot
                _input_df = _input_df.drop(axis='index', labels=n)
                deleted_count += 1
            elif self.remove_censored and status == 0 and _snapshot.total_offset() < self.forecast_horison:  # removing censored data
                _input_df = _input_df.drop(axis='index', labels=n)
                deleted_count += 1


        logging.info(f"Attention! {deleted_count} rows removed because of short OR not actual working period")

        if _input_df.empty:
            snapshot_dataset = None
        else:
            snapshot_dataset = self.fill_common_features(_common_features, _input_df, snapshot_dataset)
            # if _snapshot.num > 0:
            #     snapshot_dataset['status'] = [0 for i in range(len(snapshot_dataset['status']))]
            #     full_dataset['status'] = [0 for i in range(len(full_dataset['status']))]

            if True or 'status' not in snapshot_dataset.columns:  # client didn't provide status info
                snapshot_dataset = self.fill_status(snapshot_dataset)
            if 'age' not in snapshot_dataset.columns:  # client didn't provide age info
                snapshot_dataset = self.age_from_birth_date(snapshot_dataset)
            if True or 'seniority' not in snapshot_dataset.columns:  # client didn't provide company seniority info
                snapshot_dataset = self.seniority_from_term_date(snapshot_dataset)

            snapshot_dataset['total_seniority'] = snapshot_dataset['seniority']

            snapshot_dataset['start_date'] = np.maximum(snapshot_dataset['recruitment_date'], self.data_begin_date)
            snapshot_dataset['end_date'] = np.minimum(snapshot_dataset['termination_date'], self.data_load_date)
            snapshot_dataset['snapshot_start'] = self.calc_snapshot_start_vectorized(snapshot_dataset, _snapshot).dt.date

            # Convert to datetime64 for the calculation
            snapshot_dataset['status'] = np.where(
                (pd.to_datetime(snapshot_dataset['termination_date']) -
                 pd.to_datetime(snapshot_dataset['snapshot_start'])).dt.days / 30 > self.forecast_horison,
                0,
                snapshot_dataset['status']
            )

        return snapshot_dataset, full_dataset


    def salary_by_city(self, _dataset: pd.DataFrame):
        logging.info('Adding salary data from HeadHunter...')
        salary_df = pd.read_excel('data_raw/salary_by_cities.xlsx', sheet_name='Sheet1', index_col=0)
        _dataset['salary_by_city'] = _dataset.apply(
            lambda row: salary_df.loc[row['city'], row['job_category']]
            if row['city'] in salary_df.index and row['job_category'] in salary_df.columns
            else 80000,
            axis=1
        )
        return _dataset

    def vacations_by_city(self, _dataset: pd.DataFrame):
        logging.info('Adding vacations data from HeadHunter...')
        vacations_df = pd.read_excel('data_raw/salary_by_cities.xlsx', sheet_name='Sheet2', index_col=0)
        _dataset['vacations_by_city'] = _dataset.apply(
            lambda row: vacations_df.loc[row['city'], row['job_category']]
            if row['city'] in vacations_df.index and row['job_category'] in vacations_df.columns
            else 10,
            axis=1
        )
        return _dataset

    @classmethod
    def rubles_to_gold(self, _salary: int, _date: date):
        gold_rate = pd.read_excel('data_raw/Курс.xlsx')
        values = gold_rate.loc[gold_rate['Год'] == _date.year].drop(columns='Год').values
        value = values[0][_date.month - 1]

        return _salary / value

    def money_to_gold(self, _dataset: pd.DataFrame, _snapshot: SnapShot):
        for n, row in _dataset.iterrows():
            code = row['code']
            for col in row:
                if 'income' in col.name or 'salary' in col.name:  # money related features
                    time_point = _dataset[_dataset['code']==code]['snapshot_start']
                    income_to_gold = self.rubles_to_gold(col[0], time_point)
                    _dataset[col.name, n] = income_to_gold
        return _dataset


    def calibrate_by_inflation(self, _salary: int, _date: date, _rate: pd.DataFrame):
        if _date.year < 2022:
            print(_date)

            _date = date(year=2022, month=1, day=1)
        elif _date.year > 2024:
            print(_date)

            _date = date(year=2024, month=12, day=1)

        values = _rate.loc[_rate['Год'] == _date.year].drop(columns='Год').values

        value = values[0][_date.month - 1]
        return _salary * value

    def calibrate_by_living_wage(self, _salary: int, _date: date, _gold_rate: pd.DataFrame):
        values = _gold_rate.loc[_gold_rate['Год'] == _date.year].drop(columns='Год').values
        value = values[0][_date.month - 1]
        return _salary / value

    def calibrate_income(self, _dataset: pd.DataFrame, _snapshot: SnapShot):
        logging.info('Calibrate income by inflation...')
        rate = pd.read_excel('data_raw/Инфляция_накопительная.xlsx', sheet_name='Sheet3')
        gold_rate = pd.read_excel('data_raw/Прожиточный минимум.xlsx', sheet_name=self.region)

        for n, row in _dataset.iterrows():
            code = row['code']
            for col in _dataset.columns:
                if 'income' in col or 'salary' in col:  # money related features
                    time_point = _dataset[_dataset['code']==code]['snapshot_start'].item()
                    if self.calibrate_by == 'Inflation':
                        income_corrected = self.calibrate_by_inflation(row[col], time_point, rate)
                    elif self.calibrate_by == 'Living wage':
                        income_corrected = self.calibrate_by_living_wage(row[col], time_point, gold_rate)
                    row[col] = income_corrected
            _dataset.loc[n] = row
        return _dataset


    def check_nan_values(self, _dataset: pd.DataFrame):
        nan_rows_n = 0
        for n, row in _dataset.iterrows():
            if row.isnull().values.any():
                nan_rows_n += 1
                #_dataset = _dataset.drop(index=n)

        if nan_rows_n > 0:
            logging.info(f"{nan_rows_n} rows containing NaN values in snapshot dataset")
        return _dataset

    def run(self):
        data_file_path = os.path.join(self.data_dir, self.filename)
        data_file = pd.ExcelFile(data_file_path)
        self.check_required_sheets(self.required_sheets, data_file.sheet_names)

        input_df_common = data_file.parse(sheet_name='Основные данные')

        #uniq_regions = input_df_common['Регион'].unique()
        #for u in uniq_regions:
        #   print(u)

        snapshot = SnapShot(self.snapshot_step, self.snapshot_min_dur, self.snapshot_initial_offset, 0)
        united_dataset = pd.DataFrame()
        while True:
            logging.info(f'Starting snapshot {snapshot.num}')
            main_dataset, _ = self.collect_main_data(self.common_features_name_mapping, input_df_common, self.data_config, snapshot)
            if main_dataset is None:
                logging.info(f'No data left for snapshot {snapshot.num}. Finishing process...')
                break
            logging.debug(f'Formed main part of dataset: {main_dataset}')
            full_snapshot_dataset = self.fill_snapshot_specific(self.specific_features, data_file_path, main_dataset, snapshot)

            full_snapshot_dataset = self.check_nan_values(full_snapshot_dataset)

            full_snapshot_dataset = self.salary_by_city(full_snapshot_dataset)
            full_snapshot_dataset = self.vacations_by_city(full_snapshot_dataset)
            full_snapshot_dataset = self.calibrate_income(full_snapshot_dataset, snapshot)

            # assign employee codes with current snapshot mark:
            _dataset = self.apply_snapshot_specific_codes(full_snapshot_dataset, snapshot.num)

            logging.debug(f'Final dataset for snapshot {snapshot.num}: {full_snapshot_dataset}')
            if snapshot.num == 0:
                united_dataset = full_snapshot_dataset
            else:
                united_dataset = pd.concat([united_dataset, full_snapshot_dataset], axis=0)
            snapshot.num += 1

            if snapshot.num == self.max_snapshots_number:
                break

        # rows_to_drop = []
        # for n, row in united_dataset.iterrows():
        #     if row['status'] == 0:
        #         rows_to_drop.append(n)
        #     # row['status'] = (row['termination_date'] - self.sample_presets[int(row['code'].split('_')[0])].snapshot_start).days
        #     # united_dataset.loc[n] = row
        # united_dataset = united_dataset.drop(axis='index', labels=rows_to_drop)

        united_dataset = united_dataset.sort_values(by='code')  # sorting by code helps further splitting to train/val, keeping snapshots of each person in the same set to prevent data leakage
        # united_dataset = united_dataset.drop(columns=['recruitment_date', 'termination_date', 'code', 'birth_date'], errors='ignore')

        united_dataset.insert(0, 'field', self.get_industry_label())
        cols = united_dataset.columns.tolist()
        cols = sorted(cols)
        cols.remove('status')
        cols = cols + ['status']  # put 'status' to the end

        united_dataset = united_dataset[cols]
        # united_dataset = united_dataset.reset_index()
        logging.debug(f'\nFinal dataset:\n{united_dataset}')
        united_dataset.to_csv(self.output_path, index=False)
        logging.info(f'Saved dataset to {self.output_path}')


if __name__ == '__main__':
    pass
