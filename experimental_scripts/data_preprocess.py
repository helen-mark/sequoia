"""
  Created on Jan 2025
@author: Elena Markova
          for Attrition Rate Project
"""

import csv


def prepare_csv(_input_path: str, _output_path: str):
    # input: path to csv with raw data
    # output: csv file with data prepared for training
    file = open(_input_path, 'r')

    csv_file = csv.reader(file)

    # put in order and print non-numeric variables for better understanding of data:
    values_by_columns = {}
    title = []
    columns_of_interest = [3, 5, 6, 8, 13]

    for n, row in enumerate(csv_file):
        if n == 0:
            for col_num, val in enumerate(row):
                if col_num in columns_of_interest:
                    title.append(val)
                    values_by_columns[col_num] = {}
            print(f"title: {title}")
            continue
        for col_num, val in enumerate(row):
            if col_num in columns_of_interest:
                if val in values_by_columns[col_num].keys():
                    values_by_columns[col_num][val] += 1
                else:
                    values_by_columns[col_num][val] = 1

    for key in values_by_columns:
        print(values_by_columns[key])  # see what kinds of non-numeric values we have
    # block end

    # put in right order and write required columns:
    columns_X = [3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]  # columns currently used as independent variables
    column_Y = 5  # column currently used as target variable
    with open(_output_path, 'w') as output_file:
        writer = csv.writer(output_file, delimiter=',')
        file.seek(0)
        for row_num, row in enumerate(csv_file):
            new_row = []
            for col_num, val in enumerate(row):
                if col_num in columns_X:
                    new_row.append(val)
            new_row.append(row[column_Y])
            print(new_row)
            writer.writerow(new_row)
    # block end



if __name__ == '__main__':
    raw_data_path = 'data/Retirement Anonymized2.csv'
    final_dataset_path = 'data/dataset.csv'  #

    prepare_csv(raw_data_path, final_dataset_path)