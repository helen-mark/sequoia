"""
  Created on Jan 2025
@author: Elena Markova
          for Attrition Rate Project
"""

import os

os.environ['YDATA_LICENSE_KEY'] = '97d0ae93-9dfc-4c2a-9183-a0420a4d0771'

import pickle
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from openpyxl.drawing.image import Image
import openpyxl
from datetime import date
from pandas import read_csv
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve, balanced_accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
import optuna
from catboost import CatBoostClassifier, Pool
from sklearn.feature_selection import RFECV
from io import BytesIO
from openpyxl import Workbook


from pathlib import Path
# from examples.local import setting_dask_env

import seaborn as sn
import matplotlib.pyplot as plt

from utils.dataset_chemistry import create_features_for_datasets, collect_datasets, minority_class_resample, prepare_dataset_2, get_united_dataset, remove_short_service

# industry_avg_income = df.groupby('field')['income_shortterm'].mean().to_dict()
# df['industry_avg_income'] = df['field'].map(industry_avg_income)
# df['income_vs_industry'] = df['income_shortterm'] - df['industry_avg_income']
# position_median_income = df.groupby('department')['income_shortterm'].median().to_dict()
# df['position_median_income'] = df['department'].map(position_median_income)


def train_xgboost_classifier(_x_train, _y_train, _x_test, _y_test, _sample_weight, _num_iters):
    model = xgb.XGBClassifier(objective="binary:logistic", random_state=42, early_stopping_rounds=15, eval_set=[(_x_test, _y_test)])
    best_f1 = 0.
    best_model = model
    test_result = {}

    print(f"Fitting XGBoost classifier...")
    for iter in range(_num_iters):
        model.fit(_x_train, _y_train, eval_set=[(_x_test, _y_test)], verbose=False)
        predictions = model.predict(_x_test)

        test_result['F1'] = f1_score(_y_test, predictions)
        if test_result['F1'] > best_f1:
            best_f1 = test_result['F1']
            best_model = model

            test_result['Recall'] = recall_score(_y_test, predictions)
            test_result['Precision'] = precision_score(_y_test, predictions)

    feature_importance = best_model.get_booster().get_score(importance_type='gain')
    keys = list(feature_importance.keys())

    # Debug print:
    values = list(feature_importance.values())
    for k, v in zip(keys, values):
        print(k)
    for k, v in zip(keys, values):
        print(v)

    data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by="score", ascending=False)
    l = data.nlargest(20, columns="score").plot(kind='barh', figsize=(20, 10))  # plot top 20 features
    print(f"XGBoost test result: Recall = {test_result['Recall']}\nPrecision = {test_result['Precision']}\nF1 = {test_result['F1']}")

    feature_importance = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': _x_train.columns, 'Importance': feature_importance})
    print(feature_importance_df)

    return best_model

def train_catboost(_x_train, _y_train, _x_test, _y_test, _sample_weight, _cat_feats_encoded, _num_iters):
    # model already initialized with latest version of optimized parameters for our dataset
    model = CatBoostClassifier(
        iterations=766,  # default to 1000
        # learning_rate=0.0254,
        # od_type='Iter',
        l2_leaf_reg=12,
        bootstrap_type='MVS',
        # od_wait=52,
        depth=10,  # normally set it from 6 to 10
        eval_metric='BalancedAccuracy',
        # #random_seed=42,
        # sampling_frequency='PerTreeLevel',
        random_strength=1,  # default: 1
        # loss_function="Logloss",
        # #bagging_temperature=2,
        # auto_class_weights='Balanced',
        scale_pos_weight=2
    )


    # model = CatBoostClassifier(
    #     iterations=892,  # 400  # Fewer trees + early stopping
    #     learning_rate=0.166,  # 0.08,  # Smaller steps for better generalization
    #     depth=6,  # Slightly deeper but not excessive
    #     l2_leaf_reg=12,  # 10,  # Stronger L2 regularization
    #     bootstrap_type='MVS',
    #     # bagging_temperature=1,  # Less aggressive subsampling
    #     random_strength=1,  # Default randomness
    #     loss_function='Logloss',
    #     eval_metric='AUC',
    #     scale_pos_weight=14.512,
    #     # auto_class_weights='Balanced',  # Adjust for class imbalance
    #     od_type='IncToDec',  # Early stopping
    #     od_wait=31,  # Patience before stopping
    #     silent=True
    # )

    def objective(trial):
        train_pool = Pool(_x_train, _y_train)
        eval_pool = Pool(_x_test, _y_test)
        params={'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'iterations': trial.suggest_int('iterations', 200, 1000),
                'depth': trial.suggest_int('depth', 4, 8),
                'l2_leaf_reg': trial.suggest_int('l2_leaf_reg1', 3, 15),
                'od_wait': trial.suggest_int('od_wait', 20, 80),
                'od_type': trial.suggest_categorical('od_type', ['Iter', 'IncToDec']),
                #'bagging_temperature': trial.suggest_float('bagging_temperature', 0.5, 1),
                'random_strength': trial.suggest_int('random_strength', 1, 3),
                'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'MVS']),
                'scale_pos_weight': trial.suggest_float("scale_pos_weight", 1, 15),
                'eval_metric': 'BalancedAccuracy'}
        model1 = CatBoostClassifier(**params, verbose=0)
        model1.fit(
            _x_train, _y_train,
            eval_set=(_x_test, _y_test),
            early_stopping_rounds=50,
        )
        # Get validation predictions
        y_pred = model1.predict(_x_test)
        score = balanced_accuracy_score(_y_test, y_pred)
        #score = cross_val_score(model, pd.concat([_x_train, _x_test]), pd.concat([_y_train, _y_test]), cv=3, scoring='roc_auc').mean()
        return score
    # study = optuna.create_study(direction='maximize')
    # study.optimize(objective, n_trials=300)
    # print(f"Best parameters of optuna: {study.best_params}")

    # perform feature selection
    selector = RFECV(
        estimator=model,
        step=1,
        cv=3,
        scoring='roc_auc'
    )
    #selector.fit(_x_train, _y_train)

    #selected_features = _x_train.columns[selector.support_]
    #print(f'Selected features: {selected_features}')

    model.fit(
        _x_train,
        _y_train,
        #eval_set=(_x_test, _y_test),
        verbose=False,
        # sample_weight=_sample_weight,
        # plot=True,
        # cat_features=_cat_feats_encoded - do this if haven't encoded cat features
    )
    # with open('model_chemistry_52_49_066.pkl', 'rb') as file:
    #     model = pickle.load(file)
    train_pool = Pool(data=_x_train, label=_y_train)
    shap_values = model.get_feature_importance(prettified=False, type='ShapValues', data=train_pool)
    feature_names = _x_train.columns

    # Словарь для переименования фич
    feature_name_mapping = {
        'salary_vs_city': 'Доход / средняя з/п в городе',
        'age_group_46-55': 'Возрастная группа 46-55',
        'job_category_производство': 'Категория "Работник производства"',
        'age_sqr': 'Квадрат возраста',
        'position_industry_управление_3': 'Категория "Управление"',
        'age': 'Возраст',
        'income_group_medium': 'Уровень дохода: средний',
        'income_group_low': 'Уровень дохода: низкий',
        'income_group_medium_high': 'Уровень дохода: средне-высокий',
        'vacation_days_shortterm': 'Дни неотгуленного отпуска',
        'seniority': 'Стаж в компании',
        'income_vs_industry': 'Доход / средний доход по отрасли',
        'income_vs_position': 'Доход / средний доход по позиции',
        'income_per_experience': 'Доход / стаж',
        'position_median_income': 'Медианный доход на данной позиции',
        'income_shortterm': 'Средний доход (3 месяца)',
        'log_seniority': 'Логарифм стажа в компании',
        'age_group_36-45': 'Возрастная группа 36-45',
        'total_seniority': 'Общий стаж (трудовая книжка)',
        'absenteeism_shortterm': 'Отсутствия (3 месяца)',
        'absences_per_experience': 'Отсутствия / общий стаж',
        'absences_per_year': 'Отсутствия за год',
        'job_category_производство высококвалифицированное': 'Категория "Высококвалифицированное производство"',
        'vacations_by_city': 'Число актуальных вакансий в городе',
        'city_Невинномысск': 'Город: Невинномысск',
        'education_income_1_high': 'Высокий доход + высшее образование',
        'education_income_1_medium': 'Средний доход + высшее образование',
        'citizenship_gender_0_1': 'Женщина, российское гражданство',
        'gender_1': 'Женский пол',
        'city_population': 'Население города',
        'gender_age_0_18-25': 'Мужчина, 18-25 лет',
        'gender_age_0_26-35': 'Мужчина, 26-35 лет',
        'age_group_18-25': 'Возраст 18-25',
        'age_group_26-35': 'Возраст 26-35',
        'age_group_56-65': 'Возраст 56-65',
        'education_income_3_medium': 'Образование неизвестно, доход средний',
        'education_income_3_medium_low': 'Образование неизвестно, доход средне-низкий',
        'education_income_1_low': 'Низкий доход + высшее образование',
        'education_income_2_high': 'Высокий доход + сред. спец. образование',
        'education_income_2_low': 'Низкий доход + сред. спец. образование',
        'education_income_2_medium': 'Средний доход + сред. спец. образование',
        'education_income_3_low': 'Образование неизвестно, доход низкий',
        'citizenship_gender_0_0': 'Мужчина, российское гражданство',
        'gender_age_1_46-55': 'Женщина, 46-55 лет',
        'gender_age_1_18-25': 'Женщина, 18-25 лет',
        'income_group_high': 'Уровень дохода: высокий',
        'unused_vacation_per_experience': 'Неотгуленный отпуск / стаж',
        'children': 'число детей',
        'education_2': 'Среднее специальное образование',
        'education_1': 'Высшее образование'
    }

    # Функция для переименования фич
    def translate_feature_name(feature_name):
        return feature_name_mapping.get(feature_name, feature_name)

    # Вычисляем важность (средние SHAP значения)
    importance = shap_values[:, :-1].mean(axis=0)
    importance_abs = abs(shap_values[:, :-1]).mean(axis=0)

    # Сортировка для первого графика (знаковые значения)
    sorted_idx_sign = np.argsort(np.abs(importance))  # Индексы от наименьшего к наибольшему
    sorted_features_sign = [translate_feature_name(feature_names[i]) for i in sorted_idx_sign]
    sorted_shap_sign = [importance[i] for i in sorted_idx_sign]

    print("Feature Importance (signed values):")
    for f, i in zip(sorted_features_sign, sorted_shap_sign):
        print(f"{f}: {i:.6f}")

    # Создаем график для знаковых значений
    colors = ['red' if val > 0 else 'blue' for val in sorted_shap_sign]
    plt.figure(figsize=(12, 10))
    plt.barh(sorted_features_sign, [abs(s) for s in sorted_shap_sign], color=colors)
    plt.title('Важность признаков CatBoost (знаковые значения)', fontsize=14)
    plt.xlabel('Среднее абсолютное значение SHAP')
    plt.tight_layout()
    #plt.show()
    plt.clf()

    # Сортировка для второго графика (абсолютные значения) - топ 25
    sorted_idx_abs = np.argsort(importance_abs)[::-1]  # Индексы от наибольшего к наименьшему

    # Берем только топ 25
    top_25_idx = sorted_idx_abs[:25]
    top_25_features = [translate_feature_name(feature_names[i]) for i in top_25_idx]
    top_25_abs_shap = [importance_abs[i] for i in top_25_idx]

    # Создаем график для абсолютных значений (топ 25)
    plt.figure(figsize=(12, 10))
    plt.barh(top_25_features, top_25_abs_shap, color='skyblue')
    plt.gca().invert_yaxis()  # Чтобы самый важный признак был сверху
    plt.title('Топ 25 самых важных признаков (абсолютные значения)', fontsize=14)
    plt.xlabel('Среднее абсолютное значение SHAP')
    plt.tight_layout()
    plt.show()
    plt.clf()

    # Дополнительно: выводим топ 25 фичей
    print("\nТоп 25 самых важных признаков:")
    for i, (feature, importance_val) in enumerate(zip(top_25_features, top_25_abs_shap), 1):
        print(f"{i:2d}. {feature}: {importance_val:.6f}")

    return model


def train_random_forest_regression(_x_train, _y_train, _x_test, _y_test, _sample_weight, _num_iters):
    model = RandomForestRegressor(n_estimators=100)
    best_precision = 0.
    best_model = model

    test_result = {}
    print(f"Fitting Random Forest...")
    for iter in range(_num_iters):
        model.fit(_x_train, _y_train, sample_weight=_sample_weight)
        predictions = model.predict(_x_test)

        test_result['r2_score'] = r2_score(_y_test, predictions)
        if test_result['r2_score'] > best_precision:
            best_precision = test_result['r2_score']
            best_model = model
            for i, y in enumerate(_y_test.values):
                print(_y_test.values[i], predictions[i])

    print(f"\nRandom Forest best result: R2 score = {test_result['r2_score']}\n")

    feature_importance = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': _x_train.columns, 'Importance': feature_importance})
    print(feature_importance_df)

    return best_model

def train_random_forest_regr(_x_train, _y_train, _x_test, _y_test, _sample_weight, _num_iters):
    model = RandomForestRegressor(n_estimators=100, max_features='sqrt')
    best_precision = 0.
    best_model = model

    print("Model attr:", model.__dict__)
    test_result = {}
    print(f"Fitting Random Forest...")
    for iter in range(_num_iters):
        model.fit(_x_train, _y_train, sample_weight=_sample_weight)
        print("\nModel attr after fitting:", model.__dict__)
        predictions = model.predict(_x_test)

        # Transform probabilities to binary classification output in order to calc metrics:
        thrs = 0.5
        for i, p in enumerate(predictions):
            if p > thrs:
                predictions[i] = 1
            else:
                predictions[i] = 0
        print(_y_test, predictions)
        test_result['Precision'] = precision_score(_y_test, predictions)
        if test_result['Precision'] > best_precision:
            best_precision = test_result['Precision']
            best_model = model

            test_result['R2'] = r2_score(_y_test, predictions)
            test_result['Recall'] = recall_score(_y_test, predictions)
            test_result['F1'] = f1_score(_y_test, predictions)

    print(f"\nRandom Forest best result: R2 score = {test_result['R2']}\nRecall = {test_result['Recall']}\nPrecision = {test_result['Precision']}\nF1 = {test_result['F1']}")

    feature_importance = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': _x_train.columns, 'Importance': feature_importance})
    print(feature_importance_df)

    return best_model


def train_random_forest_cls(_x_train, _y_train, _x_test, _y_test, _sample_weight, _num_iters):
    model = RandomForestClassifier(n_estimators=100,
                                   max_depth=3,
                                   max_features=1,
                                   min_samples_leaf=7,
                                   min_samples_split=7)
    best_f1 = 0.
    best_model = model

    test_result = {}
    # grid_space = {'max_depth': [3, 5, 10, None],
    #               'n_estimators': [50, 100, 200],
    #               'max_features': [1, 3, 5, 7, 9],
    #               'min_samples_leaf': [1, 2, 3, 7],
    #               'min_samples_split': [1, 2, 3, 7]
    #               }
    # grid = GridSearchCV(model, param_grid=grid_space, cv=3, scoring='f1')
    # model_grid = grid.fit(_x_train, _y_train)
    # print('Best hyperparameters are: ' + str(model_grid.best_params_))
    # print('Best score is: ' + str(model_grid.best_score_))

    print(f"Fitting Random Forest classifier...")
    for iter in range(_num_iters):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(_x_train, _y_train)  #, sample_weight=_sample_weight)
        predictions = model.predict(_x_test)

        test_result['Precision'] = precision_score(_y_test, predictions)
        test_result['Recall'] = recall_score(_y_test, predictions)
        test_result['F1'] = f1_score(_y_test, predictions)

        if test_result['F1'] > best_f1:
            best_f1 = test_result['F1']
            best_model = model

            test_result['Recall'] = recall_score(_y_test, predictions)
            test_result['Precision'] = precision_score(_y_test, predictions)

    print(f"\nRandom Forest classifier best result: Recall = {test_result['Recall']}\nPrecision = {test_result['Precision']}\nF1 = {test_result['F1']}")

    feature_importance = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': _x_train.columns, 'Importance': feature_importance})
    print(feature_importance_df)

    return best_model



def train(_x_train, _y_train, _x_test, _y_test, _sample_weight, _cat_feats_encoded, _model_name, _num_iters):
    if _model_name == 'XGBoostClassifier':
        model = train_xgboost_classifier(_x_train, _y_train, _x_test, _y_test, _sample_weight, _num_iters)
    elif _model_name == 'RandomForestRegressor':
        model = train_random_forest_regr(_x_train, _y_train, _x_test, _y_test, _sample_weight, _num_iters)
    elif _model_name == 'RandomForestRegressor_2':
        model = train_random_forest_regression(_x_train, _y_train, _x_test, _y_test, _sample_weight, _num_iters)
    elif _model_name == "RandomForestClassifier":
        model = train_random_forest_cls(_x_train, _y_train, _x_test, _y_test, _sample_weight, _num_iters)
    elif _model_name == "CatBoostClassifier":
        model = train_catboost(_x_train, _y_train, _x_test, _y_test, _sample_weight, _cat_feats_encoded, _num_iters)
    else:
        print("Model name error: this model is not implemented yet!")
        return

    return model


def normalize(_data: pd.DataFrame):
    return _data.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)


def prepare_dataset(_dataset: pd.DataFrame, _test_split: float, _normalize: bool):
    target_idx = -1  # index of "works/left" column

    dataset = _dataset.transpose()
    trg = dataset[target_idx:]
    trn = dataset[:target_idx]

    # val_size = 2000
    # trn = trn.transpose()
    # trg = trg.transpose()
    # x_train = trn[val_size:]
    # x_test = trn[:val_size]
    # y_train = trg[val_size:]
    # y_test = trg[:val_size]

    x_train, x_test, y_train, y_test = train_test_split(trn.transpose(), trg.transpose(), test_size=_test_split)

    if _normalize:  # normalization is NOT needed for decision trees!
        x_train = normalize(x_train)
        x_test  = normalize(x_test)

    return x_train, x_test, y_train, y_test


def show_decision_tree(_model):
    tree_ = _model.estimators_[0].tree_
    feature_list = ["job_category_encoded",
                    "job_category_encoded2",
                    "job_category_encoded3",
                    "seniority_encoded",
                    "nationality_encoded",
                    "nationality_encoded2",
                    "nationality_encoded3",
                    "age_encoded",
                    "gender_encoded",
                    "gender_encoded2",
                    "gender_encoded3",
                    # "vacation_days_encoded",
                    "days_before_salary_increase_encoded",
                    "salary_increase_encoded",
                    "overtime_encoded",
                    "family_status_encoded",
                    "family_status_encoded2",
                    "family_status_encoded3",
                    "family_status_encoded4",
                    # "km_to_work_encoded",
                    "salary_6m_average_encoded",
                    "salary_cur_encoded"
                    ]
    feature_names = [feature_list[i] for i in tree_.feature]
    feature_name = [
        feature_names[i] for i in tree_.feature
    ]

    def recurse(node, depth):
        indent = "  " * depth
        if (tree_.feature[node] == -2):
            print("{}return {}".format(indent, tree_.value[node]))
        else:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)

    recurse(0, 1)


def create_shap_barchart(feature_names, shap_values, top_n=5):
    """Create a bar chart of top N SHAP values and return as image bytes"""
    # Get top N features by absolute SHAP value
    shap_series = pd.Series(shap_values, index=feature_names)
    top_shap = shap_series.abs().sort_values(ascending=False).head(top_n)

    # Create the bar chart
    fig, ax = plt.subplots(figsize=(6, 3))
    colors = ['red' if x < 0 else 'green' for x in shap_series[top_shap.index]]
    bars = ax.barh(range(len(top_shap)), shap_series[top_shap.index], color=colors)
    ax.set_yticks(range(len(top_shap)))
    ax.set_yticklabels([f"{name[:20]}..." if len(name) > 20 else name for name in top_shap.index])
    ax.set_xlabel('SHAP Value')
    ax.set_title('Top SHAP Features')
    plt.tight_layout()

    # Save to bytes buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)

    return buf, top_shap.index.tolist(), [shap_series[feature] for feature in top_shap.index]

def test(_model, _test_data):
    test_data_all = pd.concat(_test_data, axis=0)
    trg = test_data_all['status']
    feat = test_data_all.drop(columns=['status', 'code'], errors='ignore')

    N_1 = len([y for y in trg if y == 1])
    N_0 = len([y for y in trg if y == 0])

    def adjusted_precision(P, N, M, new_N, new_M):
        numerator = P * (new_N / N)
        denominator = numerator + (1 - P) * (new_M / M)
        return numerator / denominator

    predictions = _model.predict_proba(feat)
    y_proba = predictions[:, 1]
    precision, recall, thresholds = precision_recall_curve(trg, y_proba)

    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)

    # Находим порог с максимальным F1
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]

    threshold = [optimal_threshold]

    print(f"Testing on united data with threshold = {threshold}...")
    predictions = (predictions[:, 1] > threshold).astype(int)

    f1_united = f1_score(trg, predictions)
    recall_united = recall_score(trg, predictions)
    precision_united = precision_score(trg, predictions)
    precision_united = adjusted_precision(precision_united, N_1, N_0, 1430, 8570)
    ba = balanced_accuracy_score(trg, predictions)

    print(f"CatBoost result: F1 = {f1_united:.2f}, Recall = {recall_united:.2f}, Precision - {precision_united:.2f}, ba = {ba:.2f}")
    result = confusion_matrix(trg, predictions)
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(result, annot=True, annot_kws={"size": 16}, fmt='g')  # font size

    plt.show()
    plt.clf()

    print("Testing separately...")
    for t in _test_data:
        trg = t['status']
        trn = t.drop(columns=['status', 'code'])
        N_1 = len([y for y in trg if y == 1])
        N_0 = len([y for y in trg if y == 0])

        # find opti thesh:
        predictions = _model.predict_proba(trn)
        y_proba = predictions[:, 1]
        precision, recall, thresholds = precision_recall_curve(trg, y_proba)

        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)

        # Находим порог с максимальным F1
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = 0.3  # thresholds[optimal_idx]
        predictions = (predictions[:, 1] > optimal_threshold).astype(int)
        f1 = f1_score(trg, predictions)
        r = recall_score(trg, predictions)
        p = precision_score(trg, predictions)
        p = adjusted_precision(p, N_1, N_0, 1600, 8400)
        ba = balanced_accuracy_score(trg, predictions)

        print(f"test on {len(trg)} samples: thrs = {optimal_threshold}, F1={f1:.2f}, Recall={r:.2f}, Precision={p:.2f}, ba={ba:.2f}")

        predicted_classes = predictions  # (np.array(predictions) > 0.5).astype(int)
        result = confusion_matrix(trg, predicted_classes, )
        sn.set(font_scale=1.4)  # for label size
        sn.heatmap(result, annot=True, annot_kws={"size": 16}, fmt='g')  # font size

        #plt.show()
    plt.clf()

    print('Apply rowwise...')
    # results = []
    # _dataset = _test_data[0]
    # # Create lists to store results
    # results = []
    #
    # # Create a new Excel workbook
    # wb = Workbook()
    # ws = wb.active
    # ws.title = "Predictions"
    #
    # # Write headers - including columns for top SHAP features
    # headers = ['Code', 'Termination Date', 'Prediction Probability', 'Prediction Class']
    # shap_headers = [f'Top_{i + 1}_Feature' for i in range(5)] + [f'Top_{i + 1}_SHAP' for i in range(5)]
    # all_headers = headers + shap_headers + ['SHAP Chart']
    #
    # for col_idx, header in enumerate(all_headers, 1):
    #     ws.cell(row=1, column=col_idx, value=header)
    #
    # row_idx = 2  # Start from row 2
    #
    # for n, row in _dataset.iterrows():
    #     if row['status'] == 0:
    #         code = row['code']
    #         if code not in pd.read_excel('himik_predictions.xlsx')['Code'].values:
    #             continue
    #         term_date = 11
    #
    #         # Create sample with the same features as batch processing
    #         sample = _dataset.loc[_dataset['code'] == code]
    #         feat = sample.drop(columns=['status', 'code'])
    #
    #         # Get prediction probability
    #         pred_proba = _model.predict_proba(feat)[0, 1]
    #
    #         # Apply the same threshold as batch processing
    #         prediction = 1 if pred_proba > optimal_threshold else 0
    #
    #         # Write basic data to Excel for ALL predictions
    #         ws.cell(row=row_idx, column=1, value=code)
    #         ws.cell(row=row_idx, column=2, value=term_date)
    #         ws.cell(row=row_idx, column=3, value=pred_proba)
    #         ws.cell(row=row_idx, column=4, value=prediction)
    #
    #         # Only create charts for predictions == 1
    #         if prediction == 1:
    #             # Get SHAP values
    #             train_pool = Pool(data=feat, label=sample['status'])
    #             shap_values = _model.get_feature_importance(prettified=False, type='ShapValues', data=train_pool)
    #
    #             # Extract SHAP values (excluding base value)
    #             shap_for_prediction = shap_values[0, :-1]
    #             feature_names = feat.columns.tolist()
    #
    #             # Create SHAP bar chart and get top features
    #             chart_buffer, top_features, top_shap_values = create_shap_barchart(feature_names, shap_for_prediction,
    #                                                                                top_n=5)
    #
    #             # Write top SHAP features and values
    #             for i, (feature, shap_val) in enumerate(zip(top_features, top_shap_values), 1):
    #                 ws.cell(row=row_idx, column=4 + i, value=feature)  # Feature name
    #                 ws.cell(row=row_idx, column=4 + 5 + i, value=shap_val)  # SHAP value
    #
    #             # Insert SHAP bar chart
    #             img = Image(chart_buffer)
    #             img.anchor = f'{openpyxl.utils.get_column_letter(4 + 10 + 1)}{row_idx}'  # Column after SHAP values
    #             img.width = 300
    #             img.height = 150
    #             ws.add_image(img)
    #         else:
    #             # For negative predictions, you might want to add some placeholder or explanation
    #             ws.cell(row=row_idx, column=5, value="No chart - prediction = 0")
    #
    #         row_idx += 1  # Always increment row index
    #
    # # Adjust column widths
    # for col in range(1, 20):  # Adjust first 19 columns
    #     ws.column_dimensions[openpyxl.utils.get_column_letter(col)].width = 15
    #
    # # Save the workbook
    # output_file = 'himik_prediction_detailed.xlsx'
    # wb.save(output_file)
    # print(f"Detailed predictions with SHAP charts saved to {output_file}")

    return f1_united, recall_united, precision_united

def calc_weights(_y_train: pd.DataFrame, _y_val: pd.DataFrame):
    w_0, w_1 = 0, 0
    print(_y_train)
    for i in _y_train.values:
        w_0 += 1 - i.item()
        w_1 += i.item()

    tot = w_0 + w_1
    w_0 = w_0 / tot
    w_1 = w_1 / tot
    print(f"weights:", w_0, w_1)
    #return np.array([w_0 if i == 1 else w_1 for i in _y_train.values])
    return np.array([1 if i == 1 else 1 for i in _y_train.values])

def remove_outliers(dfs):
    # Features to process
    features_to_trim = [
        'vacation_days_deriv',
        'vacation_days_shortterm',
        'absenteeism_deriv',
        'absenteeism_shortterm',
        'income_deriv',
        'income_shortterm'
    ]
    new_datasets = []
    for df in dfs:
        for f in features_to_trim:
            if f not in df.columns:
                continue
            f = [f]
        # Step 1: Drop rows with NaN in features_to_trim (critical!)
            df = df.dropna(subset=f).copy()

        # Step 2: Skip features with zero variance (avoid NaN z-scores)
        # valid_features = [f for f in features_to_trim if df[f].var() > 0]
        # if not valid_features:
        #     new_datasets.append(df)  # No features left to filter
        #     continue

        # Step 3: Calculate Z-scores only on valid features
            z_scores = np.abs(stats.zscore(df[f]))
            filtered_rows = (z_scores < 2).all(axis=1)
            df_clean = df[filtered_rows].copy().reset_index(drop=True)  # Reset index here


        # Step 4: Verify no NaN in target
        assert df_clean['status'].isna().sum() == 0, \
            f"Target has {df_clean['status'].isna().sum()} NaN values after filtering!"

        new_datasets.append(df_clean)

        # Debug info
        print(f"Original rows: {len(df)}")
        print(f"Rows after outlier removal: {len(df_clean)}")
        print(f"Rows removed: {len(df) - len(df_clean)}")
    return new_datasets


def main(_config: dict):
    data_path = _config['dataset_src']
    test_path = _config['test_src']

    # Debug:
    # trn = pd.read_csv(data_path + '/chemi_dataset_24_jun.csv')
    # tst = pd.read_csv(test_path + '/chemi_dataset_24_dec.csv')

    #mask = (trn['status'] == 1) & (pd.to_datetime(trn['termination_date']).dt.date >= date(year=2024, month=9, day=1))
    #rows_to_transfer = trn[mask].copy()

    # Remove these rows from first DataFrame
    #trn = trn[~mask].copy()

    # Add them to second DataFrame
    #tst = pd.concat([tst, rows_to_transfer], ignore_index=True)

    #mask = (tst['status'] == 1) & (pd.to_datetime(tst['termination_date']).dt.date < date(year=2024, month=9, day=1))
    #rows_to_transfer = tst[mask].copy()

    # Remove these rows from first DataFrame
    #tst = tst[~mask].copy()

    # Add them to second DataFrame
    #trn = pd.concat([trn, rows_to_transfer], ignore_index=True)

    # If I want to remove extra snapshots for status==0:
    # trn = trn[~((trn['status'] == 0) & (trn['code'].str.contains('s1')))].copy()
    # tst = tst[~((tst['status'] == 0) & (tst['code'].str.contains('s1')))].copy()

    # If I want to separate personal codes in test and train:
    #status_zero_df1 = trn[trn['status'] == 0]
    #random_sample = status_zero_df1.sample(n=min(1000, len(status_zero_df1)),
#                                           random_state=42)  # Handles cases with <1000 rows
    #selected_codes = set(random_sample['code'])

    # trn = trn.drop(random_sample.index)
    # mask_to_remove = (tst['status'] == 0) & (~tst['code'].isin(selected_codes))
    # tst = tst[~mask_to_remove]  # Keep rows where the mask is False

    # trn.to_csv(data_path + '/chemi_dataset_24_jun.csv')
    # tst.to_csv(test_path + '/chemi_dataset_24_dec.csv')
    # debug end


    datasets = collect_datasets(data_path, _config['remove_small_period'], _config['remove_invalid_deriv'])
    test_datasets = collect_datasets(test_path, _config['remove_small_period'], _config['remove_invalid_deriv'])
    all_datasets = datasets + test_datasets

    # common = pd.concat(all_datasets)
    # new_test = common.sample(n=5000, random_state=43)
    # new_train = common[~common.index.isin(new_test.index)]
    # datasets = [new_train]
    # test_datasets = [new_test]
    # all_datasets = datasets + test_datasets
    # new_test.to_csv('new_test_chemy.csv')
    # new_train.to_csv('new_train_chemy.csv')

    if _config['random_split']:
        # take random split instead of splitting by years
        concat_dataset = pd.concat(datasets + test_datasets)
        trn_dataset, tst_dataset = train_test_split(concat_dataset, test_size=7000, random_state=43)
    else:
        trn_dataset = pd.concat(datasets)
        tst_dataset = pd.concat(test_datasets)
    print(len(tst_dataset[tst_dataset['status']==1]), ' 1s in TEST')
    datasets = [trn_dataset.reset_index(drop=True)]
    test_datasets = [tst_dataset.reset_index(drop=True)]
    # trn_dataset.to_csv('data/rand_train_chem.csv')
    # tst_dataset.to_csv('data/rand_test_chem.csv')

    rand_states = [4] # range(5)  # [777, 42, 6, 1370, 5087]
    score = [0, 0, 0]

    if _config['remove_short_service']:
        datasets = remove_short_service(datasets)
        test_datasets = remove_short_service(test_datasets)
        print(1, len(test_datasets[0]))

    if _config['remove_outliers']:
        datasets = [df.copy().reset_index(drop=True) for df in datasets]
        test_datasets = [df.copy().reset_index(drop=True) for df in test_datasets]
        datasets = remove_outliers(datasets)
        test_datasets = remove_outliers(test_datasets)
        print(2, len(test_datasets[0]))

    if _config['calculated_features']:
        datasets, new_cat_feat = create_features_for_datasets(datasets)
        test_datasets, _ = create_features_for_datasets(test_datasets)
        _config['cat_features'] += new_cat_feat
        print(f"New cat features: {_config['cat_features']}")


    for split_rand_state in rand_states:
        d_train, d_val, d_test, cat_feats_encoded = prepare_dataset_2(datasets,
                                                                      test_datasets,
                                                                      _config['make_synthetic'],
                                                                      _config['encode_categorical'],
                                                                      _config['use_selected_features'],
                                                                      _config['test_split'],
                                                                      _config['cat_features'],
                                                                      split_rand_state)

        if _config['smote']:
            d_train = minority_class_resample(d_train,cat_feats_encoded)
            #d_val = minority_class_resample(d_val, cat_feats_encoded)
            #d_test = minority_class_resample(d_test, cat_feats_encoded)

        x_train, y_train, x_val, y_val = get_united_dataset(d_train, d_val, d_test)
        # x_train, x_test, y_train, y_test = prepare_dataset(dataset, config['test_split'], config['normalize'])

        print(f"X train: {x_train.shape[0]}, x_val: {x_val.shape[0]}, y_train: {y_train.shape[0]}, y_val: {y_val.shape[0]}")
        sample_weight = calc_weights(y_train, y_val)
        merged = pd.merge(
            x_train.reset_index(drop=True),
            x_val.reset_index(drop=True),
            how='inner',
            indicator=False
        )
        print(f"\n\nNumber of duplicate rows: {len(merged)}")
        duplicate_indices = x_train.index.isin(merged.index)

        # Remove duplicates from x_train and y_train
        x_train = x_train[~duplicate_indices]
        y_train = y_train[~duplicate_indices]

        print(f"x_train shape after removing duplicates: {x_train.shape}")
        print(f"y_train shape after removing duplicates: {y_train.shape}")
        # print(sample_weight)
        trained_model = train(x_train.drop(columns='code', errors='ignore'), y_train, x_val.drop(columns='code', errors='ignore'), y_val, sample_weight, cat_feats_encoded, _config['model'], _config['num_iters'])

        # with open('model_45_85_0441.pkl', 'rb') as file:
        #     trained_model = pickle.load(file)
        print('Run test on TRAIN set...')
        f1, r, p = test(trained_model, d_train)
        print('Run test on TEST set...')
        f1, r, p = test(trained_model, d_test)

        score[0] += f1
        score[1] += r
        score[2] += p

        if _config['model'] == 'RandomForestClassifier':
           show_decision_tree(trained_model)

    score = (score[0] / len(rand_states), score[1] / len(rand_states), score[2] / len(rand_states))
    print(f"Final score of cross-val: F1={score[0]:.2f}, Recall = {score[1]:.2f}, Precision={score[2]:.2f}")

    with open('model.pkl', 'wb') as f:
       print("Saving model..")
       pickle.dump(trained_model, f)


if __name__ == '__main__':
    # config_path = 'config.json'  # config file is used to store some parameters of the dataset
    config = {
        'model': 'CatBoostClassifier',  # options: 'RandomForestRegressor', 'RandomForestRegressor_2','RandomForestClassifier', 'XGBoostClassifier', 'CatBoostClassifier'
        'test_split': 0.25,  # validation/training split proportion
        'normalize': False,  # normalize input values or not
        'remove_short_service': False,
        'remove_small_period': False,
        'remove_invalid_deriv': False,
        'num_iters': 20,  # number of fitting attempts
        'dataset_src': 'data/chemistry_trn_full_merged_add_short_service_rnd',
        'test_src': 'data/chemistry_tst_full_merged_add_short_service_rnd',
        'encode_categorical': True,
        'calculated_features': False,
        'use_selected_features': False,
        'remove_outliers': False,
        'make_synthetic': None,  # options: 'sdv', 'ydata', None
        'smote': False,  # perhaps not needed for catboost and in case if minority : majority > 0.5
        'random_split': False,
        'cat_features': ['gender', 'citizenship', 'job_category', 'field', 'city', 'education', 'family_status']
    }


    main(config)

# 97d0ae93-9dfc-4c2a-9183-a0420a4d0771

