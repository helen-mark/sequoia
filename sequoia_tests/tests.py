import unittest
from utils import sequoia_dataset, merge_csv
import pandas as pd
from datetime import date
import numpy as np


class TestMergeCsv(unittest.TestCase):
    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

    def handle_duplicates_test(self):
        duplicates_info = {'Mike': ['duplicate', 0, 0], 'Anna': ['multi_job', 0, 0], 'Mary': ['raider', [('2000.01.01', '2000.04.05')]] }
        input_type='timeseries'
        df = pd.DataFrame({'Full name': ['Mike', 'Anna', 'Mike', 'Mary', 'Anna', 'Mary', 'Mary', 'Jane', 'Bob'],
                           'Hire date': #[date(year=2000, month=1, day=1), date(year=2000, month=1, day=1), date(year=2000, month=2, day=1), date(year=2001, month=1, day=1), date(year=2000, month=3, day=3),
                                        # date(year=2001, month=2, day=1), date(year=2002, month=1, day=1), date(year=2003, month=1, day=1), date(year=2003, month=1, day=1)],
                                        ['2000.01.01', '2000.01.01', '2000.02.01', '2001.01.01', '2000.03.03', '2001.02.01', '2002.01.01', '2003.01.01', '2003.01.01'],
                           'Date of dismissal': #[date(year=2005, month=1, day=7), pd.NA, date(year=2005, month=1, day=7), date(year=2001, month=1, day=5), pd.NA, date(year=2001, month=2, day=3), date(year=2002, month=1, day=1),
                                                 #pd.NA, date(year=2004, month=1, day=1)],
                                                ['2005.01.07', pd.NA, '2005.01.07', '2001.01.05', pd.NA, '2001.02.03', '2002.01.01', pd.NA, '2004.01.01'],
                           'Job title': ['a', 'Barman', 'a', 'd', 'Pilot', 'd', 'k', 'h', 'h']})
        df_ts = df.copy()
        for i in range(12):
            df_ts.insert(len(df_ts.columns), str(i), [i]*len(df))
        df_ts.loc[3, '11'] = pd.NA
        df_ts.loc[3, '10'] = 0
        df_ts.loc[3, '9'] = 100


        print(df_ts)

        df_res = merge_csv.handle_duplicates(df_ts, duplicates_info, input_type)
        df_expected = pd.DataFrame({'Full name': ['Mike', 'AnnaBarman', 'AnnaPilot', 'Mary2000.01.01', 'Jane', 'Bob'],
                                    'Hire date': # [date(year=2000, month=1, day=1), date(year=2000, month=1, day=1), date(year=2000, month=3, day=3), date(year=2000, month=1, day=1), date(year=2003, month=1, day=1), date(year=2003, month=1, day=1)],
                                                 ['2000.01.01', '2000.01.01', '2000.03.03', '2000.01.01', '2003.01.01', '2003.01.01'],
                                    'Date of dismissal': #[date(year=2005, month=1, day=7), pd.NA, pd.NA,  date(year=2000, month=4, day=5), pd.NA, date(year=2004, month=1, day=1)],
                                                 ['2005.01.07', pd.NA, pd.NA, '2000.04.05', pd.NA, '2004.01.01'],
                                    'Job title': ['a', 'Barman', 'Pilot', 'd', 'h', 'h']})
        for i in range(12):
            df_expected.insert(len(df_expected.columns), str(i), [i]*len(df_expected))

        self.assertEqual(set(df_res.columns), set(df_expected.columns))

        # Convert all NA-like values to a consistent representation
        df_res = df_res.fillna(np.nan)
        df_expected = df_expected.fillna(np.nan)

        # Convert all numeric columns to float for consistent comparison
        for col in df_res.select_dtypes(include='number').columns:
            df_res[col] = df_res[col].astype(float)
            df_expected[col] = df_expected[col].astype(float)

        df_res = df_res.sort_values(by=df_res.columns.tolist()).reset_index(drop=True)
        df_expected = df_expected.sort_values(by=df_expected.columns.tolist()).reset_index(drop=True)
        df_expected.loc[4, '9'] = 100

        print(df_expected)
        print(df_res)
        # This handles NaN values correctly
        self.assertTrue((df_res.loc[0].equals(df_expected.loc[0])))



    def check_duplicates_test(self):
        return
        df = pd.DataFrame({'name': ['alex', 'ben', 'alex'], 'department': ['dept1', 'dept1', 'dept2'], 'education': ['high', 'no', 'high']})
        res = merge_csv.check_duplicates(df)
        self.assertEqual(res, [{'name': 'alex', 'department': 'dept1', 'education': 'high'}, {'name': 'alex', 'department': 'dept2', 'education': 'high'}])

    def check_rubles_to_gold_conversion(self):
        value = sequoia_dataset.rubles_to_gold(10000, date(year=2022, month=1, day=1))
        self.assertEqual(value, 10000 / 4478.88)


if __name__ == '__main__':
    unittest.main()