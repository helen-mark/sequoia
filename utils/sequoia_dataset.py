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
        self.initial_offset = _initial_offset
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
        assert _period_months > 0
        values_reversed = _values_per_month[_values_per_month.columns[::-1]]  # this operation reverses order of columns
        values_sum = 0.
        count = 0
        tot = 0
        values = [x for x in values_reversed.values[0] if str(x).lower() != 'nan']

        # Helper function to get last day of month
        def get_last_day_of_month(date):
            if date.month == 12:
                return date.replace(year=date.year + 1, month=1, day=1) - timedelta(days=1)
            return date.replace(month=date.month + 1, day=1) - timedelta(days=1)

        def update_sum(val, sum, not_full_month, not_full_last_month, recr_day, term_day, days_in_month):
            val = val.item()
            if val is None or val == np.nan or str(val).lower() == 'nan':
                return None

            val_item = str(val).split('\xa0')
            val_corr = val_item[0] + val_item[1] if len(val_item) > 1 else val_item[0]
            val_corr = float(val_corr.replace(',', '.'))
            # print("Val corr:", val_corr)
            # Handle case where it's both first AND last month (very short employment)
            if not_full_month and not_full_last_month:
                # Employee worked only from recr_day to term_day in the same month
                days_worked = term_day - recr_day + 1  # +1 to include both start and end days
                if days_worked > 0:
                    print(
                        f' Short employment: {days_worked} days worked. Approximated {val_corr} by {val_corr * days_in_month / days_worked}')
                    val_corr = val_corr * days_in_month / days_worked
                else:
                    print(f' Warning: Negative or zero days worked: recr_day={recr_day}, term_day={term_day}')

            # Handle partial first month only (recruitment)
            elif not_full_month:
                if recr_day != 1:
                    # Employee worked from recr_day to end of month
                    days_worked = days_in_month - recr_day + 1
                    print(
                        f' First month: {days_worked} days worked. Approximated {val_corr} by {val_corr * days_in_month / days_worked}')
                    val_corr = val_corr * days_in_month / days_worked

            # Handle partial last month only (termination)
            elif not_full_last_month:
                if term_day != days_in_month:
                    # Employee worked from 1st to term_day
                    days_worked = term_day  # from day 1 to term_day inclusive
                    print(
                        f' Last month: {days_worked} days worked. Approximated {val_corr} by {val_corr * days_in_month / days_worked}')
                    val_corr = val_corr * days_in_month / days_worked

            sum += val_corr
            return sum

        # avg_val = np.ma.average(values)
        for date, val in values_reversed.items():  # same order of columns like in xlsx table
            #print(f'date:{date}, snapshot:{_snapshot_timepoint}, recr: {_recr_date}, term: {_term_date}')
            if date <= _snapshot_timepoint and date >= _recr_date.replace(day=1) and date <= _term_date.replace(day=1):  # don't take the last months of the employee! the salary is incomplete
                not_full_month = date == _recr_date.replace(day=1)
                not_full_last_month = date == _term_date.replace(day=1)

                #print(f'not full month: {not_full_month}')
                tot += 1
                #print(f'tot={tot}, period={_period_months}')
                if tot > int(_period_months):
                    #print('break')
                    break

                days_in_month = get_last_day_of_month(date).day

                #print(f'day: {_recr_date.day}')
                sum = update_sum(val, values_sum, not_full_month, not_full_last_month, _recr_date.day, _term_date.day,
                                 days_in_month)
                if sum is None:
                    continue
                values_sum = sum
                count += 1


        if count == 0:
            logging.info("Found 0 valid values in time series. NOT performing imputation...")
            # for date, val in values_reversed.items():  # same order of columns like in xlsx table
            #     if date > _recr_date and date < _term_date.replace(day=1):  # don't take first and last months of the employee! the salary is incomplete
            #         sum = update_sum(val, values_sum)
            #         if sum is None:
            #             continue
            #         values_sum = sum
            #         count += 1
            #         if count >= int(_period_months):
            #             break

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

    def extract_features_from_timeseries(self, _dataset: pd.DataFrame, _feature_df: pd.DataFrame,
                                                    _longterm_avg: pd.DataFrame, _shortterm_avg: pd.DataFrame,
                                                    _deriv_col: pd.DataFrame, _snapshot: SnapShot, _feature_name: str):

        # Merge feature data with dataset in one go
        merged_data = _dataset.merge(_feature_df, on='code', how='left', suffixes=('', '_feature'))

        # Convert ALL date columns to datetime.date to ensure consistency
        date_columns = ['snapshot_start', 'start_date', 'recruitment_date', 'termination_date']
        for col in date_columns:
            merged_data[col] = pd.to_datetime(merged_data[col]).dt.date

        # Vectorized calculations for all codes at once
        codes = merged_data['code'].values
        snapshot_timepoints = merged_data['snapshot_start'].values
        start_dates = merged_data['start_date'].values
        recr_dates = merged_data['recruitment_date'].values
        term_dates = merged_data['termination_date'].values
        #print(f'start dates: {start_dates}')
        # Calculate shortterm periods vectorized
        def months_between_dates(start_date, end_date):
            """Calculate number of calendar months between two dates (inclusive counting)"""
            return (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1

        months_from_start = np.array([months_between_dates(sd, st) for st, sd in zip(snapshot_timepoints, start_dates)])
        shortterm_periods = np.minimum(self.min_window, months_from_start)
        #print(f'periods: {shortterm_periods}')

        # Calculate past_timepoints using datetime.date operations
        past_timepoints = []
        for i in range(len(codes)):
            snapshot_tp = snapshot_timepoints[i]
            start_d = start_dates[i]
            recr_d = recr_dates[i]

            # Calculate past timepoint (3 months back)
            past_tp = self.subtract_months(snapshot_tp, self.max_window - self.min_window)

            # Calculate minimum past date. We allow going past start_d, but not past recr_d
            min_past_date = self.add_days(recr_d, int(_snapshot.min_duration * 30.4))

            # Ensure past_timepoint doesn't go before start_date + min_duration
            past_tp = max(past_tp, min_past_date)
            past_timepoints.append(past_tp)

        past_timepoints = np.array(past_timepoints)

        # Calculate time delta in years
        dts = np.array([(st - pt).days / 365.0 for st, pt in zip(snapshot_timepoints, past_timepoints)])

        # Prepare results arrays
        avg_now_arr = np.full(len(codes), None, dtype=object)
        avg_past_arr = np.full(len(codes), None, dtype=object)
        deriv_arr = np.full(len(codes), None, dtype=object)

        codes_with_none_average = []
        codes_with_empty_sample = []
        # Process each sample
        for i, code in enumerate(codes):
            # Extract sample data (salary columns)
            sample = _feature_df.loc[_feature_df['code'] == code]
            if not sample.empty:
                sample = sample.drop(columns='code')  # leave only columns with salary values

            if not sample.empty and snapshot_timepoints[i] is not None:
                # Calculate averages - pass datetime.date objects
                avg_now = self.calc_numerical_average(
                    sample, shortterm_periods[i], snapshot_timepoints[i],
                    recr_dates[i], term_dates[i]
                )
                avg_past = self.calc_numerical_average(
                    sample, shortterm_periods[i], past_timepoints[i],
                    recr_dates[i], term_dates[i]
                )

                # Calculate derivative
                if avg_now is None:
                    codes_with_none_average.append(code)

                if avg_past is None or avg_now is None:
                    deriv = None
                elif dts[i] >= 1. / 4.:  # we have at least 3 months to calculate derivative
                    deriv = (avg_now - avg_past)
                else:
                    deriv = 0  # consider time period too small

                avg_now_arr[i] = avg_now
                avg_past_arr[i] = avg_past
                deriv_arr[i] = deriv
            else:
                if sample.empty:
                    codes_with_empty_sample.append(code)

        print(f'{len(codes_with_none_average)} of personal codes have None average value')
        print(f'{len(codes_with_empty_sample)} of personal codes have empty timeseries sample')


        _shortterm_avg = pd.DataFrame({
            'code': codes,
            f'{_feature_name}_shortterm': avg_now_arr
        })
        _longterm_avg = pd.DataFrame({
            'code': codes,
            f'{_feature_name}_longterm': avg_past_arr
        })
        _deriv_col = pd.DataFrame({
            'code': codes,
            f'{_feature_name}_deriv': deriv_arr
        })

        return _shortterm_avg, _longterm_avg, _deriv_col

    def subtract_months(self, date_obj, months):
        """Subtract months from a datetime.date object"""
        year = date_obj.year
        month = date_obj.month
        day = date_obj.day

        # Subtract months
        month -= months
        while month < 1:
            month += 12
            year -= 1

        # Ensure day is valid for the new month
        try:
            return date(year, month, day)
        except ValueError:
            # If day is invalid for the month (e.g., Feb 30), use last day of month
            return date(year, month, 1) + timedelta(days=-1)

    def add_days(self, date_obj, days):
        """Add days to a datetime.date object"""
        return date_obj + timedelta(days=days)

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

        shortterm_col, longterm_col, deriv_col = self.extract_features_from_timeseries(_dataset, df, longterm_col, shortterm_col, deriv_col, _snapshot, _feature_name)
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
        logging.info(f'Processing continuous feature: {_feature_name}')

        dataset = _dataset.copy()

        # Convert string values to float (handling commas as decimals)
        dataset[_feature_name] = dataset[_feature_name].str.replace(',', '.', regex=True).astype(float)

        dataset[_feature_name] = dataset[_feature_name].abs()

        days_diff = (dataset['termination_date'] - dataset['snapshot_start']).dt.days
        years_diff = days_diff / 365

        dataset[_feature_name] = dataset[_feature_name] - years_diff

        return dataset


    def fill_snapshot_specific(self, _specific_features: list, _input_file: os.path, _dataset: pd.DataFrame, _snapshot: SnapShot):
        snapshot_columns = []

        for sheet_name, feature_name in self.time_series_name_mapping.items():
            logging.info(f"Process timeseries: {feature_name}")
            df = read_excel(_input_file, sheet_name=sheet_name)
            snapshot_columns.extend(self.process_timeseries(df, _dataset, _snapshot, sheet_name, feature_name))


        # age, overall experience, company seniority:
        for feature_name in ['age', 'seniority', 'total_seniority']:  # , 'overall_experience']:
            if feature_name in _dataset.columns:
                _dataset = self.process_continuous_features(_dataset, _snapshot, feature_name)

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
            return int(key)
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
        start_date_np = pd.to_datetime(df['start_date']).values
        end_date_np = pd.to_datetime(df['end_date']).values

        seniority = df['seniority'].values
        period_of_interest = (end_date_np - start_date_np) / np.timedelta64(30, 'D')

        enough_seniority_mask = seniority > 1. / 6.
        # Mask for employees with sufficient work history within the boundaries
        mask_long_enough = period_of_interest - snapshot.initial_offset >= snapshot.min_duration

        # Create a combined mask for employees eligible for offsets
        eligible_mask = enough_seniority_mask & mask_long_enough

        # Initialize offset_months with zeros for all employees
        offset_months = np.zeros(len(df), dtype=int)

        if self.random_snapshot:
            # Compute random initial_offset where applicable
            individual_max_offset = np.floor(period_of_interest[eligible_mask] - snapshot.min_duration).astype(int)
            valid_mask = individual_max_offset > 0

            # Apply random offsets only to eligible employees
            random_offsets = np.where(
                valid_mask,
                np.random.randint(snapshot.initial_offset, individual_max_offset + 1),
                0
            )
            offset_months[eligible_mask] = random_offsets
        else:
            fixed_offset = snapshot.num * snapshot.step + snapshot.initial_offset
            valid_mask = np.floor(period_of_interest[eligible_mask] - fixed_offset - snapshot.min_duration).astype(
                int) > 0

            # Apply fixed offsets only to eligible employees
            fixed_offsets = np.where(
                valid_mask,
                fixed_offset,
                0
            )
            offset_months[eligible_mask] = fixed_offsets

        # Apply month/year arithmetic to all employees
        month_offset = (-offset_months).astype('timedelta64[M]')

        # Apply offset
        snapshot_start = (
                end_date_np.astype('datetime64[M]')
                + month_offset
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
        # too_late_mask = snapshot_start > (end_date_np.astype('datetime64[M]') - one_month).astype('datetime64[D]')
        # if np.any(too_late_mask):
        #     logging.warning(
        #         f"Adjusting {np.sum(too_late_mask)} snapshot starts to be at least 1 month before end_date"
        #     )
        #     # Set to end_date minus 1 month
        #     snapshot_start[too_late_mask] = (
        #             end_date_np[too_late_mask].astype('datetime64[M]') - one_month
        #     ).astype('datetime64[D]')

        # Convert to date objects
        def datetime64_to_date(dt64):
            return dt64.astype('datetime64[D]').item()  # Convert to date

        # Ensure the array is datetime64 (handles mixed input)
        for i, s in enumerate(snapshot_start):
            snapshot_start[i] = s.astype('datetime64[D]').item()
        # Apply conversion
        return pd.Series(snapshot_start, index=df.index)

    def collect_main_data(self, _common_features: dict, _input_df: pd.DataFrame, _data_config: dict, _snapshot: SnapShot):
        snapshot_dataset = pd.DataFrame()


        self.data_load_date = pd.to_datetime(self.data_load_date).date()  # Convert to Python date

        # Create a working copy
        df = _input_df.copy()

        # Handle missing termination dates
        missing_term_mask = (df['Date of dismissal'].isna() |
                             (df['Date of dismissal'].astype(str).str.lower() == 'nan'))
        df.loc[missing_term_mask, 'Date of dismissal'] = self.data_load_date
        df['Status'] = (~missing_term_mask).astype(int)  # 1 for has termination date, 0 for imputed

        # Remove rows with missing recruitment dates
        df = df[df['Hire date'].notna()]

        df['Hire date'] = pd.to_datetime(df['Hire date'])
        df['Date of dismissal'] = pd.to_datetime(df['Date of dismissal'])

        # Calculate employment metrics
        df['empl_period'] = (df['Date of dismissal'] - df['Hire date']).dt.days / 30.4

        df['Hire date'] = df['Hire date'].dt.date
        df['Date of dismissal'] = df['Date of dismissal'].dt.date

        df['start_date'] = pd.to_datetime(np.maximum(df['Hire date'], self.data_begin_date))
        df['end_date'] = pd.to_datetime(np.minimum(df['Date of dismissal'], self.data_load_date))
        df['max_snapshot_duration'] = (df['end_date'] - df['start_date']).dt.days / 30.4

        # Create filter masks
        filter_mask = pd.Series(True, index=df.index)

        if self.remove_short_service:
            filter_mask &= (df['empl_period'] >= 3)
            filter_mask &= (df['empl_period'] - _snapshot.total_offset() > _snapshot.min_duration)
            filter_mask &= ((df['Date of dismissal'] - self.data_begin_date).dt.days / 30.4 -
                            _snapshot.num * _snapshot.step > _snapshot.min_duration)

        filter_mask &= (df['Hire date'] < self.data_load_date)
        filter_mask &= (df['Date of dismissal'] > self.data_begin_date)

        if _snapshot.num > 0 and not self.random_snapshot:
            filter_mask &= (df['max_snapshot_duration'] > _snapshot.total_offset())

        if self.remove_censored:
            filter_mask &= ~((df['Status'] == 0) & (_snapshot.total_offset() < self.forecast_horison))

        # Apply final filter
        deleted_count = len(df) - filter_mask.sum()
        df = df[filter_mask].drop(columns=['empl_period', 'start_date', 'end_date', 'max_snapshot_duration'])


        logging.info(f"Attention! {deleted_count} rows removed because of short OR not actual working period")

        if df.empty:
            snapshot_dataset = None
        else:
            snapshot_dataset = self.fill_common_features(_common_features, df, snapshot_dataset)
            # if _snapshot.num > 0:
            #     snapshot_dataset['status'] = [0 for i in range(len(snapshot_dataset['status']))]
            #     full_dataset['status'] = [0 for i in range(len(full_dataset['status']))]

            # if True or 'status' not in snapshot_dataset.columns:  # client didn't provide status info
            #     snapshot_dataset = self.fill_status(snapshot_dataset)
            if 'age' not in snapshot_dataset.columns:  # client didn't provide age info
                snapshot_dataset = self.age_from_birth_date(snapshot_dataset)
            if True or 'seniority' not in snapshot_dataset.columns:  # client didn't provide company seniority info
                snapshot_dataset = self.seniority_from_term_date(snapshot_dataset)

            # snapshot_dataset['total_seniority'] = snapshot_dataset['seniority']

            snapshot_dataset['start_date'] = np.maximum(snapshot_dataset['recruitment_date'], self.data_begin_date)
            snapshot_dataset['end_date'] = np.minimum(snapshot_dataset['termination_date'], self.data_load_date)


        return snapshot_dataset


    def salary_by_city(self, _dataset: pd.DataFrame):
        logging.info('Adding salary data from HeadHunter...')
        salary_df = pd.read_excel('data_raw/salary_by_cities.xlsx', sheet_name='average_salary', index_col=0)
        _dataset['salary_by_city'] = _dataset.apply(
            lambda row: salary_df.loc[row['city'], row['job_category']]
            if row['city'] in salary_df.index and row['job_category'] in salary_df.columns
            else 80000,
            axis=1
        )
        return _dataset

    def vacations_by_city(self, _dataset: pd.DataFrame):
        logging.info('Adding vacations data from HeadHunter...')
        vacations_df = pd.read_excel('data_raw/salary_by_cities.xlsx', sheet_name='total_vacancies', index_col=0)
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
                    if self.calibrate_by == 'Inflation' and not pd.isna(row[col]):
                        income_corrected = self.calibrate_by_inflation(row[col], time_point, rate)
                    elif self.calibrate_by == 'Living wage' and not pd.isna(row[col]):
                        income_corrected = self.calibrate_by_living_wage(row[col], time_point, gold_rate)
                    row[col] = income_corrected
            _dataset.loc[n] = row
        return _dataset

    def check_nan_values(self, _dataset: pd.DataFrame):
        nan_counts = _dataset.isnull().sum()
        total_nan_rows = (_dataset.isnull().any(axis=1)).sum()

        if total_nan_rows > 0:
            logging.info(f"{total_nan_rows} rows containing NaN values in snapshot dataset")
            # Log detailed NaN counts per column
            for col, count in nan_counts.items():
                if count > 0:
                    logging.debug(f"Column '{col}': {count} NaN values")

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
            main_dataset = self.collect_main_data(self.common_features_name_mapping, input_df_common, self.data_config, snapshot)
            if main_dataset is None:
                logging.info(f'No data left for snapshot {snapshot.num}. Finishing process...')
            main_dataset['snapshot_start'] = self.calc_snapshot_start_vectorized(main_dataset,
                                                                                     snapshot).dt.date

            # Convert to datetime64 for the calculation
            main_dataset['status'] = np.where(
                (pd.to_datetime(main_dataset['termination_date']) -
                 pd.to_datetime(main_dataset['snapshot_start'])).dt.days / 30.4 > self.forecast_horison,
                0,
                main_dataset['status']
            )


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
