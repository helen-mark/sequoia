from auton_survival import estimators, datasets
from auton_survival.preprocessing import Preprocessor


def fit_survival_machine():
    model = estimators.SurvivalModel(model='dsm')
    print(model)
    features, outcomes = datasets.load_dataset("SUPPORT")
    print(outcomes)
    #features = Preprocessor().fit_transform(features,
    #                                        cat_feats=['GENDER', 'ETHNICITY', 'SMOKE'],
    #                                        num_feats=['height', 'weight'])


if __name__=='__main__':
    fit_survival_machine()