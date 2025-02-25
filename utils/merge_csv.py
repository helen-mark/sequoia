import os

import pandas as pd
from datetime import date
import hashlib


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
        # print(title['cat'])
        return title['cat'].item()

    writer = pd.ExcelWriter(_path, engine='openpyxl',  if_sheet_exists='replace', mode='a')
    df['Job title'] = df['Job title'].apply(get_cat)
    df.to_excel(writer, sheet_name='Основные данные', index=False, engine='openpyxl')
    writer.close()
    return df


def create_unique_code(_path: str, _sheet_names: list):
    def generate_unique_number(fio: str):
        # Преобразуем ФИО в строку
        fio_str = fio.lower().replace(" ", "")

        # Используем алгоритм sha256 для получения уникального хэша
        hash_object = hashlib.sha256()
        hash_object.update(fio_str.encode())
        hashed_fio = hash_object.hexdigest()

        # Генерируем уникальный 6-значный номер
        unique_number = int(hashed_fio[:6], 16)
        return unique_number

    writer = pd.ExcelWriter(_path, mode='a', if_sheet_exists='replace', engine='openpyxl')
    for sn in _sheet_names:
        df = pd.read_excel(_path, sheet_name=sn)

        df['Code'] = df['Full name'].apply(generate_unique_number)
        df.to_excel(writer, sheet_name=sn, index=False, engine='openpyxl')
    writer.close()
    return df


def classify_employees(_path1: str):
    df = pd.read_excel(_path1, sheet_name='Основные данные')
    job_titles = df['Job title'].values
    unique_set = set(job_titles)
    for v in sorted(unique_set):
        print(v)


def check_absent_people(_path1: str, _path2: str):
    df1 = pd.read_excel(_path1)
    df2 = pd.read_excel(_path2)
    on_column = 'Full name'
    col1 = df1[on_column]
    col2 = df2[on_column]
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
    col_to_pop = ['№', 'Legal entity', "Department", 'Hire date', 'Date of dismissal', 'Job title', 'Code']
    on_column = 'Full name'
    df_res = pd.DataFrame()

    for dirpath, dirnames, filenames in os.walk(_path):
        if dirpath == _path:
            n = 0
            filenames = sorted(filenames, reverse=False)
            for filename in filenames:
                if n == 0:
                    print("Initial file:", filename)
                    df_res = pd.read_excel(os.path.join(dirpath, filename), sheet_name='Лист_1')
                    df_res = df_res.drop(columns=col_to_pop)

                    n += 1
                    continue
                path2 = os.path.join(dirpath, filename)
                sheet2 = pd.read_excel(path2)
                sheet2 = sheet2.drop(columns=col_to_pop)

                df_res = df_res.merge(sheet2, on=on_column, how='outer')

    # remove duplictes:
    codes = df_res[on_column].values
    for code in codes:
        sample = df_res.loc[df_res[on_column] == code]
        if len(sample) > 1:
            print("Additionally remove duplicate", sample[on_column])
            df_res = df_res.drop(df_res[df_res[on_column] == code].index)
            codes = df_res[on_column].values
    result_path = os.path.join(_res_path, _compamy_name + '_' + _time_series_name + '.xlsx')

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
    on_column = 'Full name'

    for dirpath, dirnames, filenames in os.walk(_path):
        if dirpath == _path:
            n = 0
            filenames = sorted(filenames, reverse=True)
            if _check_feature:
                filenames = [f for f in filenames if _feature_name in f]
            for filename in filenames:
                if n == 0:
                    print("Initial file:", filename)
                    df_res = pd.read_excel(os.path.join(dirpath, filename), sheet_name=_feature_name)
                    n += 1
                    continue
                print('\nMerging\n', filename)
                _path2 = os.path.join(dirpath, filename)
                df_cur = pd.read_excel(_path2, sheet_name=_feature_name)

                df_res = merge_two_tables(df_res, df_cur, on_column)


    # remove duplicates:
    codes = df_res[on_column].values
    for code in codes:
        sample = df_res.loc[df_res[on_column] == code]
        if len(sample) > 1:
            print("Additionally remove duplicate", sample[on_column])
            df_res = df_res.drop(df_res[df_res[on_column] == code].index)
            codes = df_res[on_column].values

    result_filename = _company_name +'.xlsx'
    result_path = os.path.join(_path, result_filename)
    if os.path.exists(result_path):
        writer = pd.ExcelWriter(result_path, mode='a', if_sheet_exists='error', engine='openpyxl')
    else:
        writer = pd.ExcelWriter(result_path, mode='w', engine='openpyxl')

    df_res.to_excel(writer, sheet_name=_feature_name, index=False, engine='openpyxl')
    writer.close()


def merge_firms(_path: str, _res_path: str):
    df_res = pd.DataFrame()
    on_column = 'Full name'

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

                codes = df_res[on_column].values
                for n, row in df_cur.iterrows():
                    code = (row[on_column])
                    if code not in codes:
                        df_res.loc[df_res.index.max() + 1] = row
                        print("Add new row")
                    elif code in codes:
                        print("Already exists:", code)
                        sample1 = df_res.loc[df_res[on_column] == code]
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
                        df_res.loc[df_res[on_column] == code] = sample1

    result_filename = 'result_' + _path[-1] + '.xlsx'
    result_path = os.path.join(_res_path, result_filename)
    writer = pd.ExcelWriter(result_path, mode='w', engine='xlsxwriter')
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



if __name__ == '__main__':
    time_series_names = ['Отпуска', 'Выплаты', 'Отсутствия', 'Оклады', 'Сверхурочка']
    main_dir = 'Выгрузка'
    company_names = ['Берендсен', 'Вига-65', 'Рентекс-Сервис', 'Новость', 'МатСервис_МакиСервис_КовёрСервис']
    data_dir = 'Выгрузка/МатСервис_МакиСервис_КовёрСервис'


    # check_duplicates(pd.read_excel(os.path.join(main_dir, 'all_Основные данные_dupl.xlsx')))
    # merge_tables(main_dir, 'all')
    # check_absent_people(data_dir + "/q.xlsx", data_dir + '/Сверхурочка.xlsx')

    # First merge Maki and Kover into Mat
    for name in time_series_names:
        ts_dir = os.path.join('Выгрузка/МатСервис_МакиСервис_КовёрСервис', name)
        #         merge_timeseries(ts_dir)
        for sub_folder in ['a', 'b']:
            ts_sub = os.path.join(ts_dir, sub_folder)
            # merge_firms(ts_sub, ts_dir)

    # then merge main data:
    for company_name in company_names:
        company_dir = os.path.join(main_dir, company_name)
        # merge_tables(company_dir, company_name, 'main_data')

    # then merge time series of different years:
    for time_series_name in time_series_names:
        # merge_tables(company_dir, company_name)
        for company_name in company_names:
            company_dir = os.path.join(main_dir, company_name)
            # filename = time_series_name + '.xlsx'
            # time_series_dfs.append(pd.read_excel(os.path.join(company_dir, filename)))
            sub_fold = os.path.join(company_dir, time_series_name)
            merge_timeseries(sub_fold, main_dir, company_name, time_series_name)

    # then merge timeseries of different companies:
    for dirpath, dirnames, filenames in os.walk(main_dir):
        if dirpath == main_dir:
            for time_series_name in time_series_names:
                merge_tables(main_dir, 'data_final', time_series_name, True)
                pass

    final_data_path = os.path.join(main_dir, 'data_final.xlsx')

    # create codes:
    create_unique_code(final_data_path, time_series_names)

    # Finally, apply job categories:
    job_categories_path = os.path.join(main_dir, 'job_categories.xlsx')
    # job_category(final_data_path, job_categories_path)
    # filter_job_categories(final_data_path)

    # fill empty cells with zeros for absenteism table:
    df = pd.read_excel(final_data_path, sheet_name='Сверхурочка')
    df = fill_zeros(df)
    writer = pd.ExcelWriter(final_data_path, mode='a', if_sheet_exists='replace', engine='openpyxl')
    df.to_excel(writer, sheet_name='Сверхурочка', index=False)
    writer.close()


    for dirpath, dirnames, filenames in os.walk(data_dir):
        for filename in filenames:
            # print("Working with file", filename)
            path_to_file = os.path.join(dirpath, filename)
            df = pd.read_excel(path_to_file)
            # check_duplicates(df)

    file3 = os.path.join(data_dir, 'Основные данные.xlsx')
    # classify_employees(file3)
    # count_dismissed(file3)


