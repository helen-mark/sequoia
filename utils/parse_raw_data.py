from pandas import read_excel
import pandas as pd
import os
import yaml


def check_required_fields():
    pass


def fill_basic_fields():
    pass


def calc_age():
    pass


def calc_company_seniority():
    pass


def calc_overall_experience():
    pass


def calc_license_expiration():
    pass


def calc_salary_6m_average():
    pass


def calc_salary_current():
    pass


def calc_time_since_salary_increase():
    pass


def calc_income_6m():
    pass


def calc_income_cur():
    pass


def calc_time_since_promotion():
    pass


def calc_absenteeism_6m():
    pass


def calc_absenteeism_2m():
    pass


def calc_vacation_days():
    pass


def check_leader_left():
    pass


def check_has_meal():
    pass


def check_has_insurance():
    pass


def calc_penalties_2m():
    pass


def calc_penalties_6m():
    pass


def fill_snapshot_specific():
    pass


def fill_common_features():
    pass


def check_and_parse(_config):
    data_dir = _config['data_location']['data_path']
    filename = _config['data_location']['file_name']
    file_df = read_excel(os.path.join(data_dir, filename))
    print(file_df)

    dataset = pd.DataFrame()
    pass


if __name__ == '__main__':
    setup_path = 'data_config.yaml'
    with open(setup_path) as stream:
        config = yaml.load(stream, yaml.Loader)
        for key, val in config.items():
            print(val)

    check_and_parse(config)
