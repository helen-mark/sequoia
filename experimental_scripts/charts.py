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
    filename = 'data/dataset.csv'

    file = open(filename, 'r')

    csv_file = csv.reader(file)
    # csv_file_pandas = pandas.read_csv(filename, header=0, encoding='utf8')
    # histogram(csv_file)
    distribution(file, csv_file)
    # custom_histogram(csv_file)
