from sequoia_tests import tests


if __name__ == '__main__':
    t = tests.TestMergeCsv()
    t.check_duplicates_test()
    t.handle_duplicates_test()

    t = tests.TestSequoiaDataset()
    t.setUp()
    t.test_calc_numerical_average_basic()
    t.test_process_timeseries_realistic_data()
    t.test_extract_features_from_timeseries()
    t.test_calc_numerical_average_with_nan_values()
    t.test_calc_numerical_average_with_string_numbers()
    t.test_calc_numerical_average_all_nan_values()
    #t.test_extract_features_from_timeseries_with_empty_sample()
    t.test_calc_numerical_average_with_string_dates()
    t.test_calc_numerical_average_invalid_period()
