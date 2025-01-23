from auton_survival import estimators, datasets
from auton_survival.preprocessing import Preprocessor
from pandas import read_csv


def fit_survival_machine(features, outcomes):
    model = estimators.SurvivalModel(model='dsm')
    print(model)
    features = Preprocessor().fit_transform(features,
                                            cat_feats=['department', 'nationality', 'gender', 'family_status'],
                                            num_feats=[
                    "seniority",
                    "age",
                    "days_before_salary_increase",
                    "salary_increase",
                    "overtime",
                    "salary_6m_average",
                    "salary_cur"])
    print(features)
    print(outcomes.event)

    model.fit(features, outcomes)
    predictions = model.predict_risk(features, times=[8, 12, 16])
    print(predictions)


def prepare_dataset(_data_path: str):
    dataset = read_csv(config["data_path"], delimiter=',')
    dataset = dataset.transpose()
    feats = dataset[:-2].transpose()
    outs = dataset[-2:].transpose()
    return feats, outs


if __name__ == '__main__':
    config = {
        "data_path": "data/dataset-nn-small-dsm.csv"  # dataset contains columns "event" and "time"
    }

    features, outcomes = prepare_dataset(config["data_path"])
    fit_survival_machine(features, outcomes)