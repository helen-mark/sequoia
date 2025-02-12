import os

import pandas as pd
import numpy as np
import yaml
from pandas import read_excel
from datetime import date



def merge_tables(_path1: str, _path2: str, _result_path: str):
    data_file1 = pd.read_excel(_path1, sheet_name=None)
    data_file2 = pd.read_excel(_path2, sheet_name=None)
    writer = pd.ExcelWriter(result_path, mode='w', engine='xlsxwriter')
    for name, sheet1 in data_file1.items():
        print(name)
        if name in ['ДМС', 'Обед', 'Рабочее время']:
            continue
        if name == 'Основные данные':
            codes = sheet1['ФИО / код сотрудника'].values
            print('codes:', codes)
            sheet2 = data_file2[name]
            for n, row in sheet2.iterrows():
                code = int(row['ФИО / код сотрудника'])
                if code not in codes:
                    sheet1.loc[sheet1.shape[0]] = row
        else:
            sheet2 = data_file2[name]
            sheet1 = sheet1.merge(sheet2, on='ФИО / код сотрудника', how='outer')
        sheet1.to_excel(writer, sheet_name=name, index=False)
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
    data_dir = 'data/'
    file1 = 'data.xlsx'
    file2 = 'data2.xlsx'
    result_path = 'data/merged.xlsx'
    file_path1 = os.path.join(data_dir, file1)
    file_path2 = os.path.join(data_dir, file2)
    # merge_main_data(file_path1, file_path2, result_path)
    merge_tables(file_path1, file_path2, result_path)
