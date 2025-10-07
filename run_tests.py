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