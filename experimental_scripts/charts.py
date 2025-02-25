"""
  Created on Jan 2025
@author: Elena Markova
          for Attrition Rate Project

This is an auxiliary skript to create histograms from raw input data,
so that we can see distribution of values for a given variable
"""

from matplotlib import pyplot as plt
import numpy as np
import csv
import pandas
import typing


def make_chart(_column: np.array, _title: str):
    # Plotting a basic histogram
    plt.hist(_column, bins=np.size(np.unique(_column))-1, color='skyblue', edgecolor='black')

    # Adding labels and title
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title(_title)

    # Display the plot
    plt.show()


def make_curve(_df: pandas.DataFrame, _ef1: pandas.DataFrame, _ef2, _ef3):
    term_dates = _df['Date of dismissal']
    statuses = df['Status']
    date_codes = []
    plot_data = {}
    plot_data1 = {}

    ef1 = {}
    ef2 = {}
    ef3 = {}
    _ef1 = _ef1.set_index('Год')
    _ef2 = _ef2.set_index('Год')
    _ef3 = _ef3.set_index('Год')

    for i, date in enumerate(term_dates):
        if pandas.isnull(date):
            continue
        date = date.split('.')
        year_tot = int(date[2])
        year = int(date[2][-1])
        month = int(date[1])

        f1 = _ef1.loc[year_tot].values[month-1]
        f2 = _ef2.loc[year_tot].values[month-1]
        f3 = _ef3.loc[year_tot].values[month-1]

        print(year, month, f1)
        months = year*12 - 24
        date_code = months+int(date[1])  # concatenate year and month
        date_codes.append(date_code)
        if date_code in plot_data:
            plot_data[date_code] += (1 - statuses[i])
            plot_data1[date_code] += statuses[i]
        else:
            plot_data[date_code] = (1 - statuses[i])
            plot_data1[date_code] = statuses[i]

        ef1[date_code] = f1
        ef2[date_code] = f2
        ef3[date_code] = f3


    plot_data = dict(sorted(plot_data.items()))
    plot_data1 = dict(sorted(plot_data1.items()))

    ef1 = dict(sorted(ef1.items()))
    ef2 = dict(sorted(ef2.items()))
    ef3 = dict(sorted(ef3.items()))

    plt.plot(plot_data.keys(), plot_data.values(), label='Dismissal rate')
    # plt.plot(plot_data1.keys(), plot_data1.values(), label='Working rate')

    plt.plot(ef1.keys(), ef1.values(), label='Inflation')
    plt.plot(ef2.keys(), ef2.values(), label='Unemployment')
    plt.plot(ef3.keys(), ef3.values(), label='Bank rate')

    plt.legend()

    plt.title('Dismissed')
    plt.show()




def histogram(_csv_file: csv.reader):
    column_to_plt = []
    feature_idx = 10
    title = ""
    for n, row in enumerate(_csv_file):
        if n == 0:
            title = row[feature_idx]
            continue
        column_to_plt.append(int(row[feature_idx]))

    make_chart(np.sort(column_to_plt), title)


def custom_histogram(_csv_file: csv.reader):
    feature_idx = 13
    target_status_ind = 14

    points_left = {}
    points_works = {}
    histogram_title = ""
    for n, row in enumerate(_csv_file):
        if n == 0:
            histogram_title = row[feature_idx]
            continue
        worker_status = int(row[target_status_ind])
        key = float(row[feature_idx])
        if key in points_left.keys():
            points_left[key] += worker_status
            points_works[key] += 1 - worker_status
        else:
            points_left[key] = worker_status
            points_works[key] = 1 - worker_status


    plt.xlabel(histogram_title)
    plt.ylabel('Employees')
    plt.get_current_fig_manager().set_window_title(histogram_title)
    plt.stem(points_left.keys(), points_left.values(), linefmt='red', label='Left')
    plt.stem(points_works.keys(), points_works.values(), linefmt='green', label='Work')
    plt.legend()
    plt.show()


def distribution(_file: typing.TextIO, _csv_file: csv.reader):
    feature_idx = range(0,14)
    target_status_ind = 14

    for f_idx in feature_idx:
        x_points = []
        y_points = []

        x_left = []
        x_works = []

        feature_name = ""

        tot_left = 0
        tot_works = 0
        _file.seek(0)

        for n, row in enumerate(_csv_file):
            if n == 0:
                feature_name = row[f_idx]
                continue
            feature_value = float(row[f_idx])
            if feature_value != 0.:  # eliminate zero values for some features
                x_points.append(feature_value)
                y_points.append(float(row[target_status_ind]))

                if row[target_status_ind] == '0':  # Means status is "works"
                    x_works.append(feature_value)
                    tot_works += 1
                else:               # Means status is "left"
                    x_left.append(feature_value)
                    tot_left += 1

        x_left = np.sort(x_left)
        x_works = np.sort(x_works)
        x_all = [*x_left, *x_works]

        part_left = len(x_left)

        s, m = reject_outliers(x_all)

        x_left = x_left[s[:len(x_left)]<m]
        x_works = x_works[s[part_left:]<m]

        average_left = np.mean(x_left)
        average_works = np.mean(x_works)
        median_left = np.median(x_left)
        median_works = np.median(x_works)
        print(len(x_left), len(x_works))

        print(f"{feature_name}: {average_left} average for left, {average_works} average for working\n")
        # print(median_left, median_works)

        #plt.xlabel('Unemployment Rate')
        #plt.ylabel('Employee status')
        #plt.plot(x_points, y_points, 'o')
        #plt.show()


def reject_outliers(_data, _m=2.):
    d = np.abs(_data - np.median(_data))
    mdev = np.median(d)
    s = d/mdev if mdev else np.zeros(len(d))
    return s, _m


if __name__ == '__main__':
    filename = 'data/all_data_final.xlsx'
    additional_file1 = 'data/Инфляция.xlsx'
    additional_file2 = 'data/Безработица.xlsx'
    additional_file3 = 'data/Ставка ЦБ.xlsx'

    file = open(filename, 'r')

    csv_file = csv.reader(file)
    # csv_file_pandas = pandas.read_csv(filename, header=0, encoding='utf8')
    # histogram(csv_file)
    # distribution(file, csv_file)
    # custom_histogram(csv_file)

    df = pandas.read_excel(filename, sheet_name='Основные данные')
    df1 = pandas.read_excel(additional_file1)
    df2 = pandas.read_excel(additional_file2)
    df3 = pandas.read_excel(additional_file3)

    make_curve(df, df1, df2, df3)
