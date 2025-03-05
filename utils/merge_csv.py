import os

import pandas as pd
from datetime import date
import hashlib

ON_COLUMN = 'Code'
DATA_START_DATE = date(year=2022, month=1, day=1)
DATA_LOAD_DATE = date(year=2025, month=1, day=1)

def fill_zeros(_df: pd.DataFrame):
    for n, row in _df.iterrows():
        row = [0 if pd.isnull(v) else v for v in row]
        _df.iloc[n] = row

    return _df

def check_duplicates(_df: pd.DataFrame):
    feature_name = 'Full name'
    feature_values = set()
    for n, row in _df.iterrows():
        val = row[feature_name]
        feature_values.add(val)
    n_res = 0
    n_neres = 0
    for val in feature_values:
        res = _df.loc[_df[feature_name] == val]
        if len(res) > 1:
            if len(set(res['Hire date'].values)) == 1 and len(set(res['Date of dismissal'].values)) == 1:  # person has just 1 hire date and 1 dism date
                # print(res)
                for col in _df.columns:
                    if col in ['№', 'Age', 'Seniority']:
                        continue
                    res_val = res[col]
                    if len(set(res_val.values)) > 1:
                        print(n, res_val)
                        n_res += 1
                # print("SAME")
                continue
            else:  # person have been fired and taken back:
                hire = res['Hire date'].values

                dis = res['Date of dismissal']
                # if dis.isnull().all():
                #    continue
                dis = dis.values
                # print(n, hire, dis)
                n_neres += 1
    print("Resident:", n_res, 'Not resident:', n_neres)


def filter_job_categories(_path: str):
    df = pd.read_excel(_path, sheet_name='Основные данные')
    for n, row in df.iterrows():
        if row['Job title'] == 'senior':
            df = df.drop(index=n)
    writer = pd.ExcelWriter(_path, engine='openpyxl',  if_sheet_exists='replace', mode='a')
    df.to_excel(writer, sheet_name='Основные данные',index=False, engine='openpyxl')
    writer.close()


def job_category(_path: str, _cat_path: str):
    df = pd.read_excel(_path, sheet_name='Основные данные')
    cat = pd.read_excel(_cat_path, sheet_name='Sheet1')

    def get_cat(job_title: str):
        title = cat.loc[cat['job_title'] == job_title]
        if title.empty:
            print(job_title)
        print(job_title)
        return title['cat'].item()

    df['Job title'] = df['Job title'].apply(get_cat)

    writer = pd.ExcelWriter(_path, engine='openpyxl',  if_sheet_exists='replace', mode='a')
    df.to_excel(writer, sheet_name='Основные данные', index=False, engine='openpyxl')
    writer.close()
    return df

def create_unique_code(_path: str, _sheet_names: list):
    def generate_unique_number(fio: str):
        # Преобразуем ФИО в строку
        fio = str(fio)
        fio_str = fio.lower().replace(" ", "")
        fio_str = fio.split('-')
        if len(fio_str) == 1:
            fio_str = fio_str[0]
        else:
            fio_str = '1'+fio_str[1]
        # Используем алгоритм sha256 для получения уникального хэша
        hash_object = hashlib.sha256()
        hash_object.update(fio_str.encode())
        hashed_fio = hash_object.hexdigest()

        # Генерируем уникальный 6-значный номер
        unique_number = int(hashed_fio[:6], 16)

        return  unique_number

    writer = pd.ExcelWriter(_path, mode='a', if_sheet_exists='replace', engine='openpyxl')
    for sn in _sheet_names:
        df = pd.read_excel(_path, sheet_name=sn)

        df['Code'] = df[ON_COLUMN].apply(generate_unique_number)
        df.to_excel(writer, sheet_name=sn, index=False, engine='openpyxl')
    writer.close()
    return df


def classify_employees(_path1: str):
    df = pd.read_excel(_path1, sheet_name='Основные данные')
    job_titles = df['Должность'].values
    unique_set = set(job_titles)
    for v in sorted(unique_set):
        print(v)


def check_absent_people(_path1: str, _path2: str):
    df1 = pd.read_excel(_path1)
    df2 = pd.read_excel(_path2)
    col1 = df1[ON_COLUMN]
    col2 = df2[ON_COLUMN]
    n = 0
    for name in col1.values:
        if (name not in col2.values):
            n += 1
            print(n, name)

def count_dismissed(_path: str):
    df = pd.read_excel(_path, sheet_name='Основные данные')
    statuses = df['Status'].values
    n_left = 0
    n_work = 0
    for s in statuses:
        n_left += 1-s
        n_work += s
    print(f'Works: {n_work}, left: {n_left}')


def merge_timeseries(_path: str, _res_path: str, _compamy_name: str, _time_series_name: str):
    col_to_pop = []  #   ['№ п/п', 'Итого']  # ['№', 'Legal entity', "Department", 'Hire date', 'Date of dismissal', 'Job title', 'Code']
    df_res = pd.DataFrame()

    for dirpath, dirnames, filenames in os.walk(_path):
        if dirpath == _path:
            n = 0
            filenames = sorted(filenames, reverse=False)
            for filename in filenames:
                if n == 0:
                    print("Initial file:", filename)
                    df_res = pd.read_excel(os.path.join(dirpath, filename))
                    df_res = df_res.drop(columns=col_to_pop)

                    n += 1
                    continue
                path2 = os.path.join(dirpath, filename)
                sheet2 = pd.read_excel(path2)
                sheet2 = sheet2.drop(columns=col_to_pop)

                df_res = df_res.merge(sheet2, on=ON_COLUMN, how='outer')

    # remove duplicates:
    codes = df_res[ON_COLUMN].values
    for code in codes:
        sample = df_res.loc[df_res[ON_COLUMN] == code]
        if len(sample) > 1:
            print("Additionally remove duplicate", sample[ON_COLUMN])
            df_res = df_res.drop(df_res[df_res[ON_COLUMN] == code].index)
            codes = df_res[ON_COLUMN].values
    result_path = os.path.join(_res_path, _compamy_name + '_' + _time_series_name + '.xlsx')
    print("Writing result to", result_path)
    writer = pd.ExcelWriter(result_path, mode='w', engine='xlsxwriter')
    df_res.to_excel(writer, sheet_name=_time_series_name, index=False)
    writer.close()


def merge_two_tables(_df_res: pd.DataFrame, _df_cur: pd.DataFrame, _on_column: str):
    codes = _df_res[_on_column].values

    for n, row in _df_cur.iterrows():
        code = (row[_on_column])
        if code not in codes:
            _df_res.loc[_df_res.index.max() + 1] = row

        # elif code in codes:
        #     print("Already exists:", code)
        #     sample = _df_res.loc[_df_res[_on_column] == code]
        #     hire_date_1 = sample['Hire date']
        #     hire_date_2 = row['Hire date']
        #     if hire_date_2 == 'Дата приема':
        #         continue
        #     print(hire_date_1, hire_date_2)
        #
        #     if len(hire_date_1) > 1 or hire_date_1.item() != hire_date_2:  # non-resident who resign and get hired back regularly
        #         print("Attention: dropping duplicate", row[_on_column])
        #         _df_res = _df_res.drop(_df_res[_df_res[_on_column] == code].index)
        #         codes = _df_res[_on_column].values
    return _df_res

def merge_tables(_path: str, _company_name: str, _feature_name: str, _check_feature: bool = False):
    df_res = pd.DataFrame()

    for dirpath, dirnames, filenames in os.walk(_path):
        if dirpath == _path:
            n = 0
            filenames = sorted(filenames, reverse=True)
            if _check_feature:
                filenames = [f for f in filenames if _feature_name in f]
            for filename in filenames:
                if n == 0:
                    print("Initial file:", filename)
                    df_res = pd.read_excel(os.path.join(dirpath, filename))  #, sheet_name=_feature_name)
                    n += 1
                    continue
                print('\nMerging\n', filename)
                _path2 = os.path.join(dirpath, filename)
                df_cur = pd.read_excel(_path2)  # , sheet_name=_feature_name)

                df_res = merge_two_tables(df_res, df_cur, ON_COLUMN)


    # remove duplicates:
    codes = df_res[ON_COLUMN].values
    for code in codes:
        sample = df_res.loc[df_res[ON_COLUMN] == code]
        if len(sample) > 1:
            print("Additionally remove duplicate", sample[ON_COLUMN])
            df_res = df_res.drop(df_res[df_res[ON_COLUMN] == code].index)
            codes = df_res[ON_COLUMN].values

    result_filename = _company_name +'.xlsx'
    result_path = os.path.join(_path, result_filename)
    if os.path.exists(result_path):
        writer = pd.ExcelWriter(result_path, mode='a', if_sheet_exists='error', engine='openpyxl')
    else:
        writer = pd.ExcelWriter(result_path, mode='w', engine='openpyxl')

    print("writing result to ", result_path)
    df_res.to_excel(writer, sheet_name=_feature_name, index=False, engine='openpyxl')
    writer.close()


def merge_firms(_path: str, _res_path: str):
    df_res = pd.DataFrame()

    for dirpath, dirnames, filenames in os.walk(_path):
        if dirpath == _path:
            n = 0
            filenames = sorted(filenames, reverse=True)
            for filename in filenames:
                if n == 0:
                    print("Initial file:", filename)
                    df_res = pd.read_excel(os.path.join(dirpath, filename), sheet_name='Лист_1')
                    n += 1
                    continue
                print('Merging', filename)
                _path2 = os.path.join(dirpath, filename)
                df_cur = pd.read_excel(_path2, sheet_name='Лист_1')

                codes = df_res[ON_COLUMN].values
                for n, row in df_cur.iterrows():
                    code = (row[ON_COLUMN])
                    if code not in codes:
                        df_res.loc[df_res.index.max() + 1] = row
                        print("Add new row")
                    elif code in codes:
                        print("Already exists:", code)
                        sample1 = df_res.loc[df_res[ON_COLUMN] == code]
                        print("sample before:", sample1.values)
                        sample2 = row
                        if (sample1.values[0][-12:] == row.values[-12:]).all():
                            print("Already same content")
                            continue
                        n = 0
                        for col_name in sample1.columns:
                            sample1[col_name] = sample2[col_name]
                            n += 1
                            if n == 18:
                                break
                        print('sample after', sample1.values)
                        df_res.loc[df_res[ON_COLUMN] == code] = sample1

    result_filename = 'result_' + _path[-1] + '.xlsx'
    result_path = os.path.join(_res_path, result_filename)
    writer = pd.ExcelWriter(result_path, mode='w', engine='openpyxl')
    df_res.to_excel(writer, sheet_name='Лист_1', index=False)
    writer.close()


def rename_columns(_filepath: str):
    df = pd.read_excel(_filepath, sheet_name='Зарплата')
    print(df.columns)
    year = 2024
    new_columns = ['code']
    for i in range(1, 13):
        new_columns.append(date(year, i, 1))
    df.columns = new_columns


def add_zero_lines(_time_series_path: str, _main_path: str, _feature_name: str):
    df_main = pd.read_excel(_main_path)
    ts_df = pd.read_excel(_time_series_path)

    codes_main = df_main[ON_COLUMN].unique()
    codes_ts = ts_df[ON_COLUMN].unique()

    for code in codes_main:
        if code not in codes_ts:
            # add this sample and fill with zeros
            new_row = [code] + list(0 for i in range(36))
            ts_df.loc[ts_df.index.max() + 1] = new_row

    writer = pd.ExcelWriter(_time_series_path, mode='a', engine='openpyxl')
    ts_df.to_excel(writer, sheet_name=_feature_name+'_zeros', index=False)
    writer.close()


def add_zero_vacations(_main_path: str, _feature_name: str):
    df_main = pd.read_excel(_main_path, sheet_name="Основные данные")
    ts_df = pd.read_excel(_main_path, sheet_name="Отпуск")

    codes_main = df_main[ON_COLUMN].unique()
    codes_ts = ts_df[ON_COLUMN].unique()
    add_per_month = 2.33

    for n, row in df_main.iterrows():
        code = row[ON_COLUMN]
        if code not in codes_ts:
            # add this sample and fill with zeros
            hire_date = str_to_datetime(row['Дата приема'])
            hire_year = hire_date.year
            hire_month = hire_date.month
            if hire_year < 2022:
                print(hire_year)
                continue
            hire_month += 12 * (hire_year - 2022)
            new_row = [code] + list(0 for i in range(hire_month)) + list(add_per_month*i for i in range(36-hire_month))
            ts_df.loc[ts_df.index.max() + 1] = new_row

    writer = pd.ExcelWriter(_main_path, mode='a', engine='openpyxl')
    ts_df.to_excel(writer, sheet_name=_feature_name+'_zeros', index=False)
    writer.close()


def str_to_datetime(_date_str: str):
    splt = _date_str.split('.')
    return date(int(splt[2]), int(splt[1]), int(splt[0]))


def events_to_timeseries(_path_events: str, _path2: str, _company_name: str, _feature_name: str, _check_feature: bool = False):
    df_res = pd.DataFrame(columns=[ON_COLUMN, 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sept', 'oct', 'nov', 'dec'])

    print('\nMaking time series\n', _path_events)
    df_events = pd.read_excel(_path_events)  # events (vacations)
    df_init = pd.read_excel(_path2)  # initial state
    add_per_month = 2.33  # vacation days a person earns per month in Russia
    # remove duplicates:
    codes = df_events[ON_COLUMN].unique()
    num = 0
    print(f"processing {len(codes)} unique codes")
    for code in codes:
        if pd.isnull(code):
            continue
        initial = df_init.loc[df_init[ON_COLUMN] == code]
        if initial.empty:
            continue
        vacation_left = initial['Итого'].item()
        if pd.isnull(vacation_left):
            vacation_left = 0

        substract_per_month = [0 for i in range(12)]
        samples = df_events.loc[df_events[ON_COLUMN] == code]
        for n, sample in samples.iterrows():
            start = str_to_datetime(sample['Начало'])
            end = str_to_datetime(sample['Окончание'])
            days_tot = (end - start).days
            start_month = start.month
            end_month = end.month
            if start_month == end_month:
                substract_per_month[start_month-1] += days_tot
            else:
                days_1 = 30 - start.day
                days_2 = days_tot - days_1

                substract_per_month[start_month-1] += days_1
                substract_per_month[end_month-1] += days_2

        vacation_per_month = []
        sum = vacation_left
        for i in range(12):
            sum = sum + add_per_month - substract_per_month[i]
            vacation_per_month.append(sum)

        print(code, vacation_per_month)
        new_row = [code] + vacation_per_month
        print("INDEX", df_res.index)
        df_res.loc[df_res.shape[0]+1] = new_row
        num += 1

    writer = pd.ExcelWriter(_path_events, mode='a', if_sheet_exists='replace', engine='openpyxl')
    df_res.to_excel(writer, sheet_name=_feature_name+'_ts', index=False, engine='openpyxl')
    writer.close()


def worked_less_than(_df: pd.DataFrame, _less_than_months: int=2):
    lst = []
    for n, row in _df.iterrows():
        if n == 0:
            continue
        recr_date = str_to_datetime(row['Hire date'])
        recr_date = max(recr_date, DATA_START_DATE)
        if not pd.isnull(row['Date of dismissal']):
            term_date = str_to_datetime(row['Date of dismissal'])
        else:
            term_date = DATA_LOAD_DATE
        empl_period = (term_date-recr_date).days / 30
        if empl_period < _less_than_months:
            lst.append((row['Full name'], recr_date, term_date))
            print(_df.loc[_df['Full name'] == row['Full name']])
            # print(_df.loc[_df])

    #for l in sorted(lst):
    #    print(l[0], l[1], l[2])

def run_short_employment(_main_dir: str, _company_names: list):
    for company_name in _company_names:
        company_dir = os.path.join(_main_dir, company_name)
        for dirpath, _, filenames in os.walk(company_dir):
            if dirpath == company_dir:
                n = 0
                for filename in filenames:
                    if 'Сотрудники' in filename:
                        print(f"\nProcessing file {filename}\n")
                        df = pd.read_excel(os.path.join(company_dir, filename))
                        if n == 0:
                            full_df = df
                        else:
                            full_df = pd.concat([full_df, df], axis=0)
                        n += 1
                        # worked_less_than(df)
        worked_less_than(full_df)

def merge_companies(_main_dir: str, _time_series_names: list):
    for name in _time_series_names:
        ts_dir = os.path.join(_main_dir, 'МатСервис_МакиСервис_КовёрСервис', name)
        #         merge_timeseries(ts_dir)
        for sub_folder in ['a', 'b']:
            ts_sub = os.path.join(ts_dir, sub_folder)
            # merge_firms(ts_sub, ts_dir)

def merge_main_data(_main_dir: str, _company_names: list):
    for company_name in _company_names:
        company_dir = os.path.join(_main_dir, company_name)
        # merge_tables(company_dir, company_name, 'фактический', True)


if __name__ == '__main__':
    time_series_names = ['Отпуск', 'Выплаты', 'Отсутствия', 'Сверхурочка']
    main_dir = '/home/elena/ATTRITION/sequoia/Выгрузка'
    company_names =  ['Берендсен', 'Вига-65', 'Рентекс-Сервис', 'Новость', 'МатСервис_МакиСервис_КовёрСервис']
    data_dir = 'Выгрузка'

    # run_short_employment(main_dir, company_names)
    # check_duplicates(pd.read_excel(os.path.join(main_dir, 'all_Основные данные_dupl.xlsx')))
    # merge_tables(main_dir, 'all')
    # check_absent_people(data_dir + "/q.xlsx", data_dir + '/Сверхурочка.xlsx')

    # First merge Maki and Kover into Mat
    # merge_companies(main_dir, time_series_names)

    # then merge main data:
    # merge_main_data(main_dir, company_names)

    # events_to_timeseries(os.path.join(data_dir, 'фактический отпуск 2024.xlsx'), os.path.join(data_dir, 'остатки отпуска 2024.xlsx'), 'полимер', 'Отпуск')

    # then merge time series of different years:
    for time_series_name in time_series_names:
        # merge_tables(company_dir, company_name)
        for company_name in company_names:
            company_dir = os.path.join(main_dir, company_name)
            # filename = time_series_name + '.xlsx'
            # time_series_dfs.append(pd.read_excel(os.path.join(company_dir, filename)))
            sub_fold = os.path.join(company_dir, time_series_name)
            # merge_timeseries(sub_fold, main_dir, company_name, time_series_name)

    # then merge timeseries of different companies:
    for dirpath, dirnames, filenames in os.walk(main_dir):
        if dirpath == main_dir:
            for time_series_name in time_series_names:
                # merge_tables(main_dir, 'data_final', time_series_name, True)
                pass

    # add_zero_lines(os.path.join(data_dir, 'Отпуск.xlsx'), os.path.join(data_dir, 'полимер.xlsx'),'Отпуск')
    add_zero_vacations(os.path.join(data_dir, 'полимер.xlsx'), 'Отпуск')
    # final_data_path = os.path.join(main_dir, 'data_final.xlsx')

    # create codes:
    # create_unique_code(os.path.join(data_dir, 'полимер.xlsx'), time_series_names + ['Основные данные'])

    # Finally, apply job categories:
    job_categories_path = os.path.join(main_dir, 'job_categories.xlsx')
    # job_category(os.path.join(data_dir, 'полимер.xlsx'), job_categories_path)
    # filter_job_categories(final_data_path)

    # fill empty cells with zeros for absenteism table:
    # df = pd.read_excel(os.path.join(data_dir, 'полимер.xlsx'), sheet_name='Отсутствия')
    # df = fill_zeros(df)
    # writer = pd.ExcelWriter(os.path.join(data_dir, 'полимер.xlsx'), mode='a', if_sheet_exists='replace', engine='openpyxl')
    # df.to_excel(writer, sheet_name='Отсутствия', index=False)
    # writer.close()


    for dirpath, dirnames, filenames in os.walk(data_dir):
        for filename in filenames:
            # print("Working with file", filename)
            path_to_file = os.path.join(dirpath, filename)
            df = pd.read_excel(path_to_file)
            check_duplicates(df)

    file3 = os.path.join(data_dir, 'полимер.xlsx')
    # classify_employees(file3)
    # count_dismissed(file3)


