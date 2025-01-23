from auton_survival import estimators, datasets
from auton_survival.preprocessing import Preprocessor, Scaler
from auton_survival.metrics import survival_regression_metric
from pandas import read_csv


def fit_survival_machine(features, outcomes):
    model = estimators.SurvivalModel(model='dsm')
    # Preprocessor does both imputing and scaling of data:
    features = Preprocessor().fit_transform(features,
                                            cat_feats=['department', 'nationality', 'gender', 'family_status'],
                                            num_feats=[
                    "age",
                    "days_before_salary_increase",
                    "salary_increase",
                    "overtime",
                    "salary_6m_average",
                    "salary_cur"])

    print("Fitting...")
    model.fit(features, outcomes)
    print("Prediction...")
    predictions = model.predict_survival(features, times=[2000])
    print(predictions)

    # metrics = ['brs', 'ibs', 'auc', 'ctd']
    # score = survival_regression_metric(metric='brs', outcomes,
    #                                    outcomes, predictions,
    #                                    times=[20])


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