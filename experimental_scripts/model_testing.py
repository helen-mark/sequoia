import pandas as pd
from sklearn.metrics import r2_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve, balanced_accuracy_score
import numpy as np

from clusterize_by_leafs import LeafClusterThresholdOptimizer
from clusterize_by_k_means import FeatureClusterThresholdOptimizer, MixedDataClusterOptimizer


def test_with_clusters_weighted_mixed(_model, _test_data, _cat_features,
                                         feature_optimizer=None,
                                         comparison_metric='balanced_accuracy',
                                         _inference=False):
    """
    Тестирование модели с учетом кластерных порогов (MixedDataClusterOptimizer).

    :param _model: обученная модель CatBoost
    :param _test_data: список DataFrame с тестовыми данными
    :param _cat_features: список категориальных признаков
    :param feature_optimizer: предобученный MixedDataClusterOptimizer
    :param comparison_metric: метрика для сравнения ('balanced_accuracy', 'f1')
    :param _inference: флаг инференса
    """

    pos_class = 700
    neg_class = 5500

    test_data_all = pd.concat(_test_data, axis=0)
    trg = test_data_all['status']
    feat = test_data_all.drop(columns=['status', 'code'], errors='ignore')

    N_1 = len([y for y in trg if y == 1])
    N_0 = len([y for y in trg if y == 0])

    def adjusted_precision(P, N, M, new_N, new_M):
        numerator = P * (new_N / N)
        denominator = numerator + (1 - P) * (new_M / M)
        return numerator / denominator

    def find_optimal_threshold_for_cluster(y_true, y_proba, metric='balanced_accuracy'):
        """Находит оптимальный порог для кластера по указанной метрике."""
        from sklearn.metrics import balanced_accuracy_score, f1_score

        if len(np.unique(y_true)) < 2:
            return 0.5, 0.0

        # Диапазон поиска с учетом перцентилей
        proba_range = np.percentile(y_proba, [10, 90])
        thresholds = np.linspace(max(0.1, proba_range[0]),
                                 min(0.9, proba_range[1]), 50)

        best_threshold = 0.5
        best_score = 0.0

        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)

            if metric == 'balanced_accuracy':
                score = balanced_accuracy_score(y_true, y_pred)
            elif metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            else:
                raise ValueError(f"Неизвестная метрика: {metric}")

            if score > best_score:
                best_score = score
                best_threshold = threshold

        return best_threshold, best_score

    def calculate_weighted_metrics(results_dict, metric_key):
        """Рассчитывает средневзвешенную метрику"""
        if not results_dict:
            return None, None, None

        total_weight = sum(r['weight'] for r in results_dict.values())
        weighted_sum = sum(r[metric_key] * r['weight'] for r in results_dict.values())
        weighted_avg = weighted_sum / total_weight if total_weight > 0 else 0

        # Также посчитаем простое среднее для сравнения
        simple_avg = np.mean([r[metric_key] for r in results_dict.values()])

        # Медиана (не взвешенная)
        median_val = np.median([r[metric_key] for r in results_dict.values()])

        return weighted_avg, simple_avg, median_val

    # Словари для хранения результатов
    category_results = {}
    cluster_results = {}

    # ==================== ТЕСТИРОВАНИЕ ПО КАТЕГОРИЯМ ====================
    cat_columns = [col for col in feat.columns if col.startswith('job_category_2_')]
    for cat_col in cat_columns:
        cat_name = cat_col.replace('job_category_2_', '')

        for seniority in [100]:
            mask = (feat[cat_col] == 1) & (feat['seniority'] < seniority)

            if mask.sum() > 100:
                predictions = _model.predict_proba(feat[mask])
                y_proba_united = predictions[:, 1]
                trg_filtered = trg[mask]

                if len(np.unique(trg_filtered)) > 1:
                    precision, recall, thresholds = precision_recall_curve(trg_filtered, y_proba_united)
                    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
                    optimal_idx = np.argmax(f1_scores)
                    optimal_threshold = thresholds[optimal_idx]

                    predictions_bin = (y_proba_united > optimal_threshold).astype(int)

                    f1_united = f1_score(trg_filtered, predictions_bin)
                    recall_united = recall_score(trg_filtered, predictions_bin)
                    precision_united = precision_score(trg_filtered, predictions_bin)
                    ba = balanced_accuracy_score(trg_filtered, predictions_bin)

                    category_results[cat_name] = {
                        'f1': f1_united,
                        'recall': recall_united,
                        'precision': precision_united,
                        'ba': ba,
                        'threshold': optimal_threshold,
                        'size': mask.sum(),
                        'weight': mask.sum() / len(feat),
                        'positive_rate': trg_filtered.mean(),
                        'pred_positive_rate': predictions_bin.mean()
                    }

    # ==================== ТЕСТИРОВАНИЕ ПО КЛАСТЕРАМ ====================
    if feature_optimizer is not None:
        print(f'\n{"=" * 60}')
        print("ТЕСТИРОВАНИЕ ПО КЛАСТЕРАМ (СМЕШАННЫЕ ДАННЫЕ)")
        print(f'{"=" * 60}')

        # Проверяем, что оптимизатор обучен - ИСПРАВЛЕННАЯ ПРОВЕРКА!
        if not hasattr(feature_optimizer, 'is_fitted'):
            print("⚠️  У оптимизатора нет атрибута is_fitted!")
            # Попробуем проверить другими способами
            if hasattr(feature_optimizer, 'cluster_info') and feature_optimizer.cluster_info:
                print("✅ Но есть cluster_info, продолжаем...")
                feature_optimizer.is_fitted = True  # Устанавливаем вручную
            else:
                print("❌ Оптимизатор не обучен! Пропускаем кластерный анализ.")
                feature_optimizer = None

        elif not feature_optimizer.is_fitted:
            print("⚠️  Оптимизатор не обучен (is_fitted=False)! Пропускаем кластерный анализ.")
            feature_optimizer = None

        if feature_optimizer is not None:
            predictions_all = _model.predict_proba(feat)
            y_proba_all = predictions_all[:, 1]

            try:
                # Получаем кластеры
                cluster_labels = feature_optimizer.predict_cluster(feat)
                unique_clusters = np.unique(cluster_labels)

                print(f"Найдено кластеров: {len(unique_clusters)}")

                # Собираем статистику по кластерам
                cluster_stats = []
                for cluster_name in unique_clusters:
                    cluster_mask = (cluster_labels == cluster_name)
                    cluster_size = cluster_mask.sum()
                    cluster_stats.append((cluster_name, cluster_size))

                # Сортируем по размеру
                cluster_stats.sort(key=lambda x: x[1], reverse=True)

                print("\nТоп-10 кластеров по размеру:")
                for i, (cluster_name, size) in enumerate(cluster_stats[:10]):
                    print(f"  {i + 1}. {cluster_name}: {size} объектов ({size / len(feat) * 100:.1f}%)")

                # Анализируем каждый кластер
                processed_clusters = 0
                for cluster_name in unique_clusters:
                    cluster_mask = (cluster_labels == cluster_name)

                    if cluster_mask.sum() > 10:  # Минимальный размер
                        y_proba_cluster = y_proba_all[cluster_mask]
                        trg_cluster = trg[cluster_mask]

                        if len(np.unique(trg_cluster)) > 1:
                            # 1. Кластерный порог (из оптимизатора)
                            cluster_threshold = feature_optimizer.cluster_info.get(cluster_name, {}).get('threshold',
                                                                                                         0.5)
                            predictions_cluster = (y_proba_cluster > cluster_threshold).astype(int)

                            # Метрики с кластерным порогом
                            if comparison_metric == 'balanced_accuracy':
                                score_cluster = balanced_accuracy_score(trg_cluster, predictions_cluster)
                            elif comparison_metric == 'f1':
                                score_cluster = f1_score(trg_cluster, predictions_cluster, zero_division=0)

                            # 2. Оптимальный порог по той же метрике
                            optimal_threshold, score_optimal = find_optimal_threshold_for_cluster(
                                trg_cluster, y_proba_cluster,
                                metric=comparison_metric
                            )
                            predictions_optimal = (y_proba_cluster > optimal_threshold).astype(int)

                            # Дополнительные метрики для информации
                            f1_cluster = f1_score(trg_cluster, predictions_cluster, zero_division=0)
                            ba_cluster = balanced_accuracy_score(trg_cluster, predictions_cluster)
                            f1_optimal = f1_score(trg_cluster, predictions_optimal, zero_division=0)
                            ba_optimal = balanced_accuracy_score(trg_cluster, predictions_optimal)

                            cluster_results[cluster_name] = {
                                'score_cluster': score_cluster,
                                'score_optimal': score_optimal,
                                'f1_cluster': f1_cluster,
                                'f1_optimal': f1_optimal,
                                'ba_cluster': ba_cluster,
                                'ba_optimal': ba_optimal,
                                'cluster_threshold': cluster_threshold,
                                'optimal_threshold': optimal_threshold,
                                'size': cluster_mask.sum(),
                                'weight': cluster_mask.sum() / len(feat),
                                'positive_rate': trg_cluster.mean(),
                                'pred_positive_rate': predictions_cluster.mean(),
                                'improvement': score_optimal - score_cluster
                            }

                            processed_clusters += 1

                print(f"\nУспешно обработано {processed_clusters} кластеров")

                # Выводим информацию о лучших/худших кластерах
                if cluster_results:
                    # Лучшие кластеры по улучшению
                    sorted_by_improvement = sorted(
                        cluster_results.items(),
                        key=lambda x: x[1]['improvement'],
                        reverse=True
                    )[:5]

                    print(f"\nТоп-5 кластеров по улучшению ({comparison_metric}):")
                    for cluster_name, metrics in sorted_by_improvement:
                        print(f"  {cluster_name}:")
                        print(f"    Размер: {metrics['size']} ({metrics['weight'] * 100:.1f}%)")
                        print(
                            f"    {comparison_metric}: {metrics['score_cluster']:.3f} -> {metrics['score_optimal']:.3f}")
                        print(f"    Улучшение: {metrics['improvement']:+.3f}")
                        print(f"    Порог: {metrics['cluster_threshold']:.3f} -> {metrics['optimal_threshold']:.3f}")

            except Exception as e:
                print(f"❌ Ошибка при предсказании кластеров: {e}")
                import traceback
                traceback.print_exc()
                cluster_results = {}

    # ==================== ВЫВОД РЕЗУЛЬТАТОВ ====================
    print(f'\n{"=" * 60}')
    print(f"СРАВНЕНИЕ ПОДХОДОВ (Метрика: {comparison_metric})")
    print(f'{"=" * 60}')

    results_summary = {}

    # Категории
    if category_results:
        # Для категорий используем BA (или можно пересчитать под comparison_metric)
        ba_weighted, ba_simple, ba_median = calculate_weighted_metrics(category_results, 'ba')
        f1_weighted, f1_simple, f1_median = calculate_weighted_metrics(category_results, 'f1')

        results_summary['categories'] = {
            'ba_weighted': ba_weighted,
            'ba_simple': ba_simple,
            'f1_weighted': f1_weighted,
            'f1_simple': f1_simple,
            'n_groups': len(category_results),
            'coverage': sum(r['size'] for r in category_results.values()) / len(feat)
        }

        print(f"\n📊 КАТЕГОРИИ (job_category_2):")
        print(f"  Групп: {len(category_results)}")
        print(f"  Покрытие данных: {results_summary['categories']['coverage'] * 100:.1f}%")
        print(f"  Balanced Accuracy:")
        print(f"    Средневзвешенная: {ba_weighted:.4f}")
        print(f"    Простое среднее:  {ba_simple:.4f}")
        print(f"  F1-score:")
        print(f"    Средневзвешенный: {f1_weighted:.4f}")
        print(f"    Простое среднее:  {f1_simple:.4f}")

    # Кластеры
    if cluster_results:
        if comparison_metric == 'balanced_accuracy':
            metric_key = 'ba_cluster'
            metric_key_optimal = 'ba_optimal'
        else:  # 'f1'
            metric_key = 'f1_cluster'
            metric_key_optimal = 'f1_optimal'

        # С кластерными порогами
        score_cluster_w, score_cluster_s, _ = calculate_weighted_metrics(
            cluster_results, 'score_cluster'
        )
        # С оптимальными порогами
        score_optimal_w, score_optimal_s, _ = calculate_weighted_metrics(
            cluster_results, 'score_optimal'
        )

        # Также BA для сравнения
        ba_cluster_w, ba_cluster_s, _ = calculate_weighted_metrics(cluster_results, 'ba_cluster')
        ba_optimal_w, ba_optimal_s, _ = calculate_weighted_metrics(cluster_results, 'ba_optimal')

        results_summary['clusters'] = {
            'score_cluster_weighted': score_cluster_w,
            'score_cluster_simple': score_cluster_s,
            'score_optimal_weighted': score_optimal_w,
            'score_optimal_simple': score_optimal_s,
            'ba_cluster_weighted': ba_cluster_w,
            'ba_cluster_simple': ba_cluster_s,
            'ba_optimal_weighted': ba_optimal_w,
            'ba_optimal_simple': ba_optimal_s,
            'n_groups': len(cluster_results),
            'coverage': sum(r['size'] for r in cluster_results.values()) / len(feat),
            'avg_improvement': np.mean([r['improvement'] for r in cluster_results.values()])
        }

        print(f"\n📊 КЛАСТЕРЫ (MixedDataClusterOptimizer):")
        print(f"  Групп: {len(cluster_results)}")
        print(f"  Покрытие данных: {results_summary['clusters']['coverage'] * 100:.1f}%")
        print(f"  Основная метрика ({comparison_metric}):")
        print(f"    С кластерными порогами:")
        print(f"      Средневзвешенная: {score_cluster_w:.4f}")
        print(f"      Простое среднее:  {score_cluster_s:.4f}")
        print(f"    С оптимальными порогами:")
        print(f"      Средневзвешенная: {score_optimal_w:.4f}")
        print(f"      Простое среднее:  {score_optimal_s:.4f}")
        print(f"    Улучшение: {score_optimal_w - score_cluster_w:+.4f}")
        print(f"    Среднее улучшение по кластерам: {results_summary['clusters']['avg_improvement']:+.4f}")

        # Balanced Accuracy для полноты картины
        print(f"\n  Balanced Accuracy (для сравнения):")
        print(f"    С кластерными порогами: {ba_cluster_w:.4f}")
        print(f"    С оптимальными порогами: {ba_optimal_w:.4f}")

    # ==================== СРАВНЕНИЕ ПОДХОДОВ ====================
    if category_results and cluster_results:
        print(f'\n{"=" * 60}')
        print("ИТОГОВОЕ СРАВНЕНИЕ КАТЕГОРИЙ И КЛАСТЕРОВ")
        print(f'{"=" * 60}')

        # Сравниваем по основной метрике
        if comparison_metric == 'balanced_accuracy':
            categories_score = results_summary['categories']['ba_weighted']
            clusters_score = results_summary['clusters']['score_optimal_weighted']
        else:  # 'f1'
            categories_score = results_summary['categories']['f1_weighted']
            clusters_score = results_summary['clusters']['score_optimal_weighted']

        improvement = clusters_score - categories_score

        print(f"\nСравнение по {comparison_metric}:")
        print(f"  Категории:    {categories_score:.4f}")
        print(f"  Кластеры:     {clusters_score:.4f}")
        print(f"  Разница:      {improvement:+.4f}")

        # Также сравниваем по BA
        categories_ba = results_summary['categories']['ba_weighted']
        clusters_ba = results_summary['clusters']['ba_optimal_weighted']
        improvement_ba = clusters_ba - categories_ba

        print(f"\nСравнение по Balanced Accuracy:")
        print(f"  Категории:    {categories_ba:.4f}")
        print(f"  Кластеры:     {clusters_ba:.4f}")
        print(f"  Разница:      {improvement_ba:+.4f}")

        # Формулируем вывод
        if improvement > 0.01:
            print(f"\n✅ Кластерный подход СУЩЕСТВЕННО улучшает {comparison_metric}!")
        elif improvement > 0:
            print(f"\n✅ Кластерный подход НЕМНОГО улучшает {comparison_metric}")
        elif improvement > -0.01:
            print(f"\n⚠️  Кластерный подход НЕ УХУДШАЕТ {comparison_metric}")
        else:
            print(f"\n❌ Кластерный подход УХУДШАЕТ {comparison_metric}")

        # Рекомендация
        if improvement > 0:
            print(f"\n🏆 РЕКОМЕНДАЦИЯ: Использовать кластерные пороги")
            print(f"   Улучшение: {improvement:+.4f} по {comparison_metric}")
        else:
            print(f"\n🏆 РЕКОМЕНДАЦИЯ: Оставить категориальные пороги")
            print(f"   Кластеры не дают улучшения")

    # ==================== ВИЗУАЛИЗАЦИЯ ====================
    if cluster_results:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            print(f'\n{"=" * 60}')
            print("ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
            print(f'{"=" * 60}')

            # Подготовка данных для графиков
            clusters_df = pd.DataFrame.from_dict(cluster_results, orient='index')
            clusters_df = clusters_df.sort_values('size', ascending=False)

            fig, axes = plt.subplots(2, 3, figsize=(18, 12))

            # 1. Размеры кластеров
            axes[0, 0].bar(range(len(clusters_df)), clusters_df['size'].values)
            axes[0, 0].set_xlabel('Кластеры (отсортированы по размеру)')
            axes[0, 0].set_ylabel('Размер кластера')
            axes[0, 0].set_title('Распределение размеров кластеров')
            axes[0, 0].tick_params(axis='x', rotation=45)

            # 2. Сравнение метрик
            axes[0, 1].scatter(range(len(clusters_df)),
                               clusters_df['score_cluster'].values,
                               alpha=0.6, label='Кластерные пороги', s=50)
            axes[0, 1].scatter(range(len(clusters_df)),
                               clusters_df['score_optimal'].values,
                               alpha=0.6, label='Оптимальные пороги', s=50)
            axes[0, 1].set_xlabel('Кластеры')
            axes[0, 1].set_ylabel(f'{comparison_metric.upper()}')
            axes[0, 1].set_title('Сравнение качества по кластерам')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # 3. Улучшение vs размер кластера
            axes[0, 2].scatter(clusters_df['size'].values,
                               clusters_df['improvement'].values,
                               alpha=0.6, s=50)
            axes[0, 2].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            axes[0, 2].set_xlabel('Размер кластера')
            axes[0, 2].set_ylabel('Улучшение')
            axes[0, 2].set_title('Улучшение vs размер кластера')
            axes[0, 2].set_xscale('log')
            axes[0, 2].grid(True, alpha=0.3)

            # 4. Распределение порогов
            axes[1, 0].hist(clusters_df['cluster_threshold'].values,
                            alpha=0.5, label='Кластерные', bins=15)
            axes[1, 0].hist(clusters_df['optimal_threshold'].values,
                            alpha=0.5, label='Оптимальные', bins=15)
            axes[1, 0].set_xlabel('Порог')
            axes[1, 0].set_ylabel('Частота')
            axes[1, 0].set_title('Распределение порогов')
            axes[1, 0].legend()
            axes[1, 0].axvline(x=0.5, color='red', linestyle='--', alpha=0.5)

            # 5. Качество vs доля положительного класса
            axes[1, 1].scatter(clusters_df['positive_rate'].values,
                               clusters_df['score_optimal'].values,
                               alpha=0.6, s=50)
            axes[1, 1].set_xlabel('Доля положительного класса')
            axes[1, 1].set_ylabel(f'{comparison_metric.upper()}')
            axes[1, 1].set_title('Качество vs дисбаланс')
            axes[1, 1].grid(True, alpha=0.3)

            # 6. Разность порогов vs улучшение
            threshold_diff = clusters_df['optimal_threshold'] - clusters_df['cluster_threshold']
            axes[1, 2].scatter(threshold_diff.values,
                               clusters_df['improvement'].values,
                               alpha=0.6, s=50)
            axes[1, 2].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            axes[1, 2].axvline(x=0, color='red', linestyle='--', alpha=0.5)
            axes[1, 2].set_xlabel('Разность порогов (оптим - кластер)')
            axes[1, 2].set_ylabel('Улучшение')
            axes[1, 2].set_title('Влияние изменения порога на качество')
            axes[1, 2].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"\n⚠️  Не удалось создать визуализации: {e}")

    return {
        'category_results': category_results,
        'cluster_results': cluster_results,
        'results_summary': results_summary,
        'comparison_metric': comparison_metric
    }


# Пример использования:
def test_with_mixed_clusters(model, train_data, test_data):
    """
    Полный пайплайн тестирования с MixedDataClusterOptimizer
    """

    # 1. Подготовка данных
    X_train = train_data.drop(columns=['status', 'code'], errors='ignore')
    y_train = train_data['status']

    X_test = [test_data] if isinstance(test_data, pd.DataFrame) else test_data

    # 2. Создаем и обучаем MixedDataClusterOptimizer
    print("Создание MixedDataClusterOptimizer...")
    mixed_optimizer = MixedDataClusterOptimizer(
        n_clusters=25,
        min_cluster_size=100,
        categorical_distance='jaccard',
        metric='balanced_accuracy',
        random_state=42
    )

    # Получаем вероятности на трейне
    y_pred_proba_train = model.predict_proba(X_train)[:, 1]

    # Обучаем кластеризатор
    print("Обучение кластеризатора...")
    mixed_optimizer.fit(
        X=X_train,
        y=y_train,
        y_pred_proba=y_pred_proba_train
    )

    # 3. Тестирование
    print("\nТестирование модели...")
    results = test_with_clusters_weighted_advanced(
        _model=model,
        _test_data=X_test,
        _cat_features=[],  # Категориальные признаки уже учтены в оптимизаторе
        feature_optimizer=mixed_optimizer,
        comparison_metric='balanced_accuracy',
        _inference=False
    )

    return results, mixed_optimizer



def test_with_clusters_weighted(_model, _test_data, _cat_features,
                                         cluster_optimizer=None,
                                         feature_optimizer=None,
                                         _inference=False):
    """
    Тестирование модели с учетом кластерных порогов (разные типы кластеризации).

    :param _model: обученная модель CatBoost
    :param _test_data: список DataFrame с тестовыми данными
    :param _cat_features: список категориальных признаков
    :param cluster_optimizer: предобученный LeafClusterThresholdOptimizer (по листьям)
    :param feature_optimizer: предобученный FeatureClusterThresholdOptimizer (по признакам)
    :param _inference: флаг инференса
    """

    pos_class = 700
    neg_class = 5500

    test_data_all = pd.concat(_test_data, axis=0)
    trg = test_data_all['status']
    feat = test_data_all.drop(columns=['status', 'code'], errors='ignore')

    N_1 = len([y for y in trg if y == 1])
    N_0 = len([y for y in trg if y == 0])

    def adjusted_precision(P, N, M, new_N, new_M):
        numerator = P * (new_N / N)
        denominator = numerator + (1 - P) * (new_M / M)
        return numerator / denominator

    # Функция для расчета средневзвешенных метрик
    def calculate_weighted_metrics(results_dict, metric_key):
        """Рассчитывает средневзвешенную метрику"""
        if not results_dict:
            return None, None, None

        total_weight = sum(r['weight'] for r in results_dict.values())
        weighted_sum = sum(r[metric_key] * r['weight'] for r in results_dict.values())
        weighted_avg = weighted_sum / total_weight if total_weight > 0 else 0

        # Также посчитаем простое среднее для сравнения
        simple_avg = np.mean([r[metric_key] for r in results_dict.values()])

        # Медиана (не взвешенная)
        median_val = np.median([r[metric_key] for r in results_dict.values()])

        return weighted_avg, simple_avg, median_val

    # Словари для хранения результатов
    category_results = {}
    leaf_cluster_results = {}
    feature_cluster_results = {}

    # ==================== ТЕСТИРОВАНИЕ ПО КАТЕГОРИЯМ ====================
    cat_columns = [col for col in feat.columns if col.startswith('job_category_2_')]
    for cat_col in cat_columns:
        cat_name = cat_col.replace('job_category_2_', '')

        for seniority in [100]:
            mask = (feat[cat_col] == 1) & (feat['seniority'] < seniority)

            if mask.sum() > 100:
                predictions = _model.predict_proba(feat[mask])
                y_proba_united = predictions[:, 1]
                trg_filtered = trg[mask]

                if len(np.unique(trg_filtered)) > 1:
                    precision, recall, thresholds = precision_recall_curve(trg_filtered, y_proba_united)
                    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
                    optimal_idx = np.argmax(f1_scores)
                    optimal_threshold = thresholds[optimal_idx]

                    predictions_bin = (y_proba_united > optimal_threshold).astype(int)

                    f1_united = f1_score(trg_filtered, predictions_bin)
                    recall_united = recall_score(trg_filtered, predictions_bin)
                    precision_united = precision_score(trg_filtered, predictions_bin)
                    ba = balanced_accuracy_score(trg_filtered, predictions_bin)

                    category_results[cat_name] = {
                        'f1': f1_united,
                        'recall': recall_united,
                        'precision': precision_united,
                        'ba': ba,
                        'threshold': optimal_threshold,
                        'size': mask.sum(),
                        'weight': mask.sum() / len(feat),
                        'positive_rate': trg_filtered.mean(),
                        'pred_positive_rate': predictions_bin.mean()
                    }

    # ==================== ТЕСТИРОВАНИЕ ПО КЛАСТЕРАМ (ЛИСТЬЯ) ====================
    if cluster_optimizer is not None:
        print(f'\n{"=" * 60}')
        print("ТЕСТИРОВАНИЕ ПО КЛАСТЕРАМ (ЛИСТЬЯ ДЕРЕВЬЕВ)")
        print(f'{"=" * 60}')

        predictions_all = _model.predict_proba(feat)
        y_proba_all = predictions_all[:, 1]
        cluster_labels = cluster_optimizer.predict_cluster(feat)
        unique_clusters = np.unique(cluster_labels)

        print(f"Найдено кластеров: {len(unique_clusters)}")

        for cluster_name in unique_clusters:
            cluster_mask = (cluster_labels == cluster_name)

            if cluster_mask.sum() > 10:
                y_proba_cluster = y_proba_all[cluster_mask]
                trg_cluster = trg[cluster_mask]

                if len(np.unique(trg_cluster)) > 1:
                    # Кластерный порог
                    cluster_threshold = cluster_optimizer.cluster_info.get(cluster_name, {}).get('threshold', 0.5)
                    predictions_cluster = (y_proba_cluster > cluster_threshold).astype(int)

                    f1_cluster = f1_score(trg_cluster, predictions_cluster)
                    ba_cluster = balanced_accuracy_score(trg_cluster, predictions_cluster)

                    # Оптимальный порог
                    precision_curve, recall_curve, thresholds_curve = precision_recall_curve(
                        trg_cluster, y_proba_cluster
                    )
                    f1_scores_curve = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-9)

                    if len(f1_scores_curve) > 0:
                        optimal_idx = np.argmax(f1_scores_curve)
                        optimal_threshold = thresholds_curve[optimal_idx]
                        predictions_optimal = (y_proba_cluster > optimal_threshold).astype(int)
                        f1_optimal = f1_score(trg_cluster, predictions_optimal)
                        ba_optimal = balanced_accuracy_score(trg_cluster, predictions_optimal)
                    else:
                        optimal_threshold = cluster_threshold
                        f1_optimal = f1_cluster
                        ba_optimal = ba_cluster

                    leaf_cluster_results[cluster_name] = {
                        'f1_cluster_thresh': f1_cluster,
                        'f1_optimal_thresh': f1_optimal,
                        'ba_cluster_thresh': ba_cluster,
                        'ba_optimal_thresh': ba_optimal,
                        'cluster_threshold': cluster_threshold,
                        'optimal_threshold': optimal_threshold,
                        'size': cluster_mask.sum(),
                        'weight': cluster_mask.sum() / len(feat),
                        'positive_rate': trg_cluster.mean(),
                        'pred_positive_rate': predictions_cluster.mean()
                    }

    # ==================== ТЕСТИРОВАНИЕ ПО КЛАСТЕРАМ (ПРИЗНАКИ) ====================
    if feature_optimizer is not None:
        print(f'\n{"=" * 60}')
        print("ТЕСТИРОВАНИЕ ПО КЛАСТЕРАМ (ПРИЗНАКИ K-MEANS)")
        print(f'{"=" * 60}')

        predictions_all = _model.predict_proba(feat)
        y_proba_all = predictions_all[:, 1]
        cluster_labels = feature_optimizer.predict_cluster(feat)
        unique_clusters = np.unique(cluster_labels)

        print(f"Найдено кластеров: {len(unique_clusters)}")

        for cluster_name in unique_clusters:
            cluster_mask = (cluster_labels == cluster_name)

            if cluster_mask.sum() > 10:
                y_proba_cluster = y_proba_all[cluster_mask]
                trg_cluster = trg[cluster_mask]

                if len(np.unique(trg_cluster)) > 1:
                    # Кластерный порог
                    cluster_threshold = feature_optimizer.cluster_info.get(cluster_name, {}).get('threshold', 0.5)
                    predictions_cluster = (y_proba_cluster > cluster_threshold).astype(int)

                    f1_cluster = f1_score(trg_cluster, predictions_cluster)
                    ba_cluster = balanced_accuracy_score(trg_cluster, predictions_cluster)

                    # Оптимальный порог
                    precision_curve, recall_curve, thresholds_curve = precision_recall_curve(
                        trg_cluster, y_proba_cluster
                    )
                    f1_scores_curve = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-9)

                    if len(f1_scores_curve) > 0:
                        optimal_idx = np.argmax(f1_scores_curve)
                        optimal_threshold = thresholds_curve[optimal_idx]
                        predictions_optimal = (y_proba_cluster > optimal_threshold).astype(int)
                        f1_optimal = f1_score(trg_cluster, predictions_optimal)
                        ba_optimal = balanced_accuracy_score(trg_cluster, predictions_optimal)
                    else:
                        optimal_threshold = cluster_threshold
                        f1_optimal = f1_cluster
                        ba_optimal = ba_cluster

                    feature_cluster_results[cluster_name] = {
                        'f1_cluster_thresh': f1_cluster,
                        'f1_optimal_thresh': f1_optimal,
                        'ba_cluster_thresh': ba_cluster,
                        'ba_optimal_thresh': ba_optimal,
                        'cluster_threshold': cluster_threshold,
                        'optimal_threshold': optimal_threshold,
                        'size': cluster_mask.sum(),
                        'weight': cluster_mask.sum() / len(feat),
                        'positive_rate': trg_cluster.mean(),
                        'pred_positive_rate': predictions_cluster.mean()
                    }

    # ==================== ВЫВОД РЕЗУЛЬТАТОВ ====================
    print(f'\n{"=" * 60}')
    print("СРАВНЕНИЕ ВСЕХ ПОДХОДОВ (СРЕДНЕВЗВЕШЕННЫЕ МЕТРИКИ)")
    print(f'{"=" * 60}')

    results_summary = {}

    # Категории
    if category_results:
        ba_weighted, ba_simple, ba_median = calculate_weighted_metrics(category_results, 'ba')
        f1_weighted, f1_simple, f1_median = calculate_weighted_metrics(category_results, 'f1')

        results_summary['categories'] = {
            'ba_weighted': ba_weighted,
            'ba_simple': ba_simple,
            'f1_weighted': f1_weighted,
            'f1_simple': f1_simple,
            'n_groups': len(category_results),
            'coverage': sum(r['size'] for r in category_results.values()) / len(feat)
        }

        print(f"\n📊 КАТЕГОРИИ (job_category_2):")
        print(f"  Групп: {len(category_results)}")
        print(f"  Покрытие данных: {results_summary['categories']['coverage'] * 100:.1f}%")
        print(f"  Balanced Accuracy:")
        print(f"    Средневзвешенная: {ba_weighted:.4f}")
        print(f"    Простое среднее:  {ba_simple:.4f}")
        print(f"  F1-score:")
        print(f"    Средневзвешенный: {f1_weighted:.4f}")
        print(f"    Простое среднее:  {f1_simple:.4f}")

    # Кластеры по листьям
    if leaf_cluster_results:
        ba_cluster_w, ba_cluster_s, _ = calculate_weighted_metrics(leaf_cluster_results, 'ba_cluster_thresh')
        ba_optimal_w, ba_optimal_s, _ = calculate_weighted_metrics(leaf_cluster_results, 'ba_optimal_thresh')

        results_summary['leaf_clusters'] = {
            'ba_cluster_weighted': ba_cluster_w,
            'ba_cluster_simple': ba_cluster_s,
            'ba_optimal_weighted': ba_optimal_w,
            'ba_optimal_simple': ba_optimal_s,
            'n_groups': len(leaf_cluster_results),
            'coverage': sum(r['size'] for r in leaf_cluster_results.values()) / len(feat)
        }

        print(f"\n📊 КЛАСТЕРЫ ПО ЛИСТЬЯМ:")
        print(f"  Групп: {len(leaf_cluster_results)}")
        print(f"  Покрытие данных: {results_summary['leaf_clusters']['coverage'] * 100:.1f}%")
        print(f"  Balanced Accuracy:")
        print(f"    С кластерными порогами:")
        print(f"      Средневзвешенная: {ba_cluster_w:.4f}")
        print(f"      Простое среднее:  {ba_cluster_s:.4f}")
        print(f"    С оптимальными порогами:")
        print(f"      Средневзвешенная: {ba_optimal_w:.4f}")
        print(f"      Простое среднее:  {ba_optimal_s:.4f}")
        print(f"    Улучшение (оптим vs кластер): {ba_optimal_w - ba_cluster_w:+.4f}")

    # Кластеры по признакам
    if feature_cluster_results:
        ba_cluster_w, ba_cluster_s, _ = calculate_weighted_metrics(feature_cluster_results, 'ba_cluster_thresh')
        ba_optimal_w, ba_optimal_s, _ = calculate_weighted_metrics(feature_cluster_results, 'ba_optimal_thresh')

        results_summary['feature_clusters'] = {
            'ba_cluster_weighted': ba_cluster_w,
            'ba_cluster_simple': ba_cluster_s,
            'ba_optimal_weighted': ba_optimal_w,
            'ba_optimal_simple': ba_optimal_s,
            'n_groups': len(feature_cluster_results),
            'coverage': sum(r['size'] for r in feature_cluster_results.values()) / len(feat)
        }

        print(f"\n📊 КЛАСТЕРЫ ПО ПРИЗНАКАМ (K-means):")
        print(f"  Групп: {len(feature_cluster_results)}")
        print(f"  Покрытие данных: {results_summary['feature_clusters']['coverage'] * 100:.1f}%")
        print(f"  Balanced Accuracy:")
        print(f"    С кластерными порогами:")
        print(f"      Средневзвешенная: {ba_cluster_w:.4f}")
        print(f"      Простое среднее:  {ba_cluster_s:.4f}")
        print(f"    С оптимальными порогами:")
        print(f"      Средневзвешенная: {ba_optimal_w:.4f}")
        print(f"      Простое среднее:  {ba_optimal_s:.4f}")
        print(f"    Улучшение (оптим vs кластер): {ba_optimal_w - ba_cluster_w:+.4f}")

    # ==================== СРАВНЕНИЕ ПОДХОДОВ ====================
    print(f'\n{"=" * 60}')
    print("ИТОГОВОЕ СРАВНЕНИЕ")
    print(f'{"=" * 60}')

    # Создаем таблицу сравнения
    comparison_data = []

    if 'categories' in results_summary:
        comparison_data.append({
            'Method': 'Categories',
            'BA_weighted': results_summary['categories']['ba_weighted'],
            'F1_weighted': results_summary['categories']['f1_weighted'],
            'N_groups': results_summary['categories']['n_groups'],
            'Coverage': results_summary['categories']['coverage']
        })

    if 'leaf_clusters' in results_summary:
        comparison_data.append({
            'Method': 'Leaf Clusters',
            'BA_weighted': results_summary['leaf_clusters']['ba_optimal_weighted'],
            'F1_weighted': None,  # Можно добавить если нужно
            'N_groups': results_summary['leaf_clusters']['n_groups'],
            'Coverage': results_summary['leaf_clusters']['coverage']
        })

    if 'feature_clusters' in results_summary:
        comparison_data.append({
            'Method': 'Feature Clusters',
            'BA_weighted': results_summary['feature_clusters']['ba_optimal_weighted'],
            'F1_weighted': None,  # Можно добавить если нужно
            'N_groups': results_summary['feature_clusters']['n_groups'],
            'Coverage': results_summary['feature_clusters']['coverage']
        })

    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        print(f"\n{comparison_df.to_string(index=False)}")

        # Определяем лучший метод
        if len(comparison_df) > 1:
            best_method = comparison_df.loc[comparison_df['BA_weighted'].idxmax(), 'Method']
            best_ba = comparison_df['BA_weighted'].max()

            print(f"\n🏆 ЛУЧШИЙ МЕТОД: {best_method}")
            print(f"   Balanced Accuracy: {best_ba:.4f}")

            # Сравнение с категориями
            if 'categories' in results_summary:
                categories_ba = results_summary['categories']['ba_weighted']
                improvement = best_ba - categories_ba

                if improvement > 0.01:
                    print(f"   ✅ Улучшение относительно категорий: {improvement:+.4f}")
                elif improvement > 0:
                    print(f"   ⚠️  Небольшое улучшение: {improvement:+.4f}")
                else:
                    print(f"   ❌ Ухудшение относительно категорий: {improvement:+.4f}")

    return {
        'category_results': category_results,
        'leaf_cluster_results': leaf_cluster_results,
        'feature_cluster_results': feature_cluster_results,
        'results_summary': results_summary,
        'comparison_df': comparison_df if 'comparison_df' in locals() else None
    }


# Пример использования
# if __name__ == "__main__":
#     # Ваш код загрузки данных
#
#     # 1. Категории (без оптимизатора)
#     results = test_with_clusters_weighted_advanced(
#         _model=model,
#         _test_data=test_data_list,
#         _cat_features=cat_features,
#         cluster_optimizer=None,
#         feature_optimizer=None
#     )
#
#     # 2. С кластерами по листьям
#     if False:  # Можете включить если хотите проверить
#         leaf_optimizer = LeafClusterThresholdOptimizer(
#             model=model,
#             n_trees=2,
#             min_cluster_size=50
#         )
#
#         y_pred_proba_train = model.predict_proba(X_train)[:, 1]
#         leaf_optimizer.fit(X_train, y_train, y_pred_proba_train)
#
#         results = test_with_clusters_weighted_advanced(
#             _model=model,
#             _test_data=test_data_list,
#             _cat_features=cat_features,
#             cluster_optimizer=leaf_optimizer,
#             feature_optimizer=None
#         )
#
#     # 3. С кластерами по признакам (РЕКОМЕНДУЕМЫЙ ВАРИАНТ)
#     feature_optimizer = FeatureClusterThresholdOptimizer(
#         n_clusters=30,  # Количество кластеров
#         min_cluster_size=50,  # Минимальный размер кластера
#         use_pca=True,  # Использовать PCA
#         pca_components=20,  # Количество компонент PCA
#         random_state=42
#     )
#
#     y_pred_proba_train = model.predict_proba(X_train)[:, 1]
#     feature_optimizer.fit(X_train, y_train, y_pred_proba_train)
#
#     results = test_with_clusters_weighted_advanced(
#         _model=model,
#         _test_data=test_data_list,
#         _cat_features=cat_features,
#         cluster_optimizer=None,
#         feature_optimizer=feature_optimizer
#     )


def test_with_leaf_clusters_weighted(_model, _test_data, _cat_features, optimizer=None, _inference=False):
    """
    Тестирование модели с учетом кластерных порогов и средневзвешенными метриками

    :param _model: обученная модель CatBoost
    :param _test_data: список DataFrame с тестовыми данными
    :param _cat_features: список категориальных признаков
    :param optimizer: предобученный LeafClusterThresholdOptimizer
    :param _inference: флаг инференса
    """

    pos_class = 700
    neg_class = 5500

    test_data_all = pd.concat(_test_data, axis=0)
    trg = test_data_all['status']
    feat = test_data_all.drop(columns=['code'], errors='ignore')

    N_1 = len([y for y in trg if y == 1])
    N_0 = len([y for y in trg if y == 0])

    def adjusted_precision(P, N, M, new_N, new_M):
        numerator = P * (new_N / N)
        denominator = numerator + (1 - P) * (new_M / M)
        return numerator / denominator

    # Словари для хранения результатов с весами
    category_results = {}
    cluster_results = {}

    # ==================== ТЕСТИРОВАНИЕ ПО КАТЕГОРИЯМ ====================
    cat_columns = [col for col in feat.columns if col.startswith('job_category_2_')]
    for cat_col in cat_columns:
        cat_name = cat_col.replace('job_category_2_', '')

        for seniority in [100]:
            # Фильтруем данные по категории и seniority
            mask = (feat[cat_col] == 1) & (feat['seniority'] < seniority)

            if mask.sum() > 100:
                # Получаем предсказания
                predictions = _model.predict_proba(feat[mask].drop(columns=['status']))
                y_proba_united = predictions[:, 1]
                trg_filtered = trg[mask]

                if len(np.unique(trg_filtered)) > 1:
                    # Подбираем оптимальный порог по F1
                    precision, recall, thresholds = precision_recall_curve(trg_filtered, y_proba_united)
                    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
                    optimal_idx = np.argmax(f1_scores)
                    optimal_threshold = thresholds[optimal_idx]

                    # Предсказания с оптимальным порогом
                    predictions_bin = (y_proba_united > optimal_threshold).astype(int)

                    # Рассчитываем метрики
                    f1_united = f1_score(trg_filtered, predictions_bin)
                    recall_united = recall_score(trg_filtered, predictions_bin)
                    precision_united = precision_score(trg_filtered, predictions_bin)
                    ba = balanced_accuracy_score(trg_filtered, predictions_bin)

                    # Сохраняем результаты с весом (размером выборки)
                    category_results[cat_name] = {
                        'f1': f1_united,
                        'recall': recall_united,
                        'precision': precision_united,
                        'ba': ba,
                        'threshold': optimal_threshold,
                        'size': mask.sum(),
                        'weight': mask.sum() / len(feat),  # Доля от общего количества
                        'positive_rate': trg_filtered.mean(),
                        'pred_positive_rate': predictions_bin.mean()
                    }

    # ==================== ТЕСТИРОВАНИЕ ПО КЛАСТЕРАМ ====================
    if optimizer is not None:
        print(f'\n{"=" * 60}')
        print("ТЕСТИРОВАНИЕ ПО КЛАСТЕРАМ")
        print(f'{"=" * 60}')

        # Получаем предсказанные вероятности для всех данных
        predictions_all = _model.predict_proba(feat.drop(columns=['status']))
        y_proba_all = predictions_all[:, 1]

        # Получаем кластеры для всех данных
        cluster_labels = optimizer.predict_cluster(feat.drop(columns=['status']))

        # Уникальные кластеры
        unique_clusters = np.unique(cluster_labels)
        print(f"Найдено кластеров: {len(unique_clusters)}")

        for cluster_name in unique_clusters:
            # Маска для текущего кластера
            cluster_mask = (cluster_labels == cluster_name)

            if cluster_mask.sum() > 10:  # Минимальный размер кластера
                # Данные кластера
                y_proba_cluster = y_proba_all[cluster_mask]
                trg_cluster = trg[cluster_mask]

                if len(np.unique(trg_cluster)) > 1:
                    # Вариант 1: Использовать порог из оптимизатора
                    cluster_threshold = optimizer.cluster_info.get(cluster_name, {}).get('threshold', 0.5)
                    predictions_cluster = (y_proba_cluster > cluster_threshold).astype(int)

                    # Метрики с кластерным порогом
                    f1_cluster = f1_score(trg_cluster, predictions_cluster)
                    ba_cluster = balanced_accuracy_score(trg_cluster, predictions_cluster)

                    # Вариант 2: Подобрать оптимальный порог по F1 для этого кластера
                    precision_curve, recall_curve, thresholds_curve = precision_recall_curve(
                        trg_cluster, y_proba_cluster
                    )
                    f1_scores_curve = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-9)
                    if len(f1_scores_curve) > 0:  # Проверка на случай ошибки
                        optimal_idx = np.argmax(f1_scores_curve)
                        optimal_threshold = thresholds_curve[optimal_idx]

                        predictions_optimal = (y_proba_cluster > optimal_threshold).astype(int)
                        f1_optimal = f1_score(trg_cluster, predictions_optimal)
                        ba_optimal = balanced_accuracy_score(trg_cluster, predictions_optimal)
                    else:
                        optimal_threshold = cluster_threshold
                        f1_optimal = f1_cluster
                        ba_optimal = ba_cluster

                    # Сохраняем результаты с весом
                    cluster_results[cluster_name] = {
                        'f1_cluster_thresh': f1_cluster,
                        'f1_optimal_thresh': f1_optimal,
                        'ba_cluster_thresh': ba_cluster,
                        'ba_optimal_thresh': ba_optimal,
                        'cluster_threshold': cluster_threshold,
                        'optimal_threshold': optimal_threshold,
                        'size': cluster_mask.sum(),
                        'weight': cluster_mask.sum() / len(feat),  # Доля от общего количества
                        'positive_rate': trg_cluster.mean(),
                        'pred_positive_rate': predictions_cluster.mean()
                    }

    # ==================== РАСЧЕТ СРЕДНЕВЗВЕШЕННЫХ МЕТРИК ====================
    print(f'\n{"=" * 60}')
    print("СРЕДНЕВЗВЕШЕННЫЕ МЕТРИКИ")
    print(f'{"=" * 60}')

    # Функция для расчета средневзвешенных метрик
    def calculate_weighted_metrics(results_dict, metric_key):
        """Рассчитывает средневзвешенную метрику"""
        if not results_dict:
            return None, None, None

        total_weight = sum(r['weight'] for r in results_dict.values())
        weighted_sum = sum(r[metric_key] * r['weight'] for r in results_dict.values())
        weighted_avg = weighted_sum / total_weight if total_weight > 0 else 0

        # Также посчитаем простое среднее для сравнения
        simple_avg = np.mean([r[metric_key] for r in results_dict.values()])

        # Медиана (не взвешенная)
        median_val = np.median([r[metric_key] for r in results_dict.values()])

        return weighted_avg, simple_avg, median_val

    # Средневзвешенные метрики по категориям
    if category_results:
        ba_weighted_cat, ba_simple_cat, ba_median_cat = calculate_weighted_metrics(
            category_results, 'ba'
        )
        f1_weighted_cat, f1_simple_cat, f1_median_cat = calculate_weighted_metrics(
            category_results, 'f1'
        )

        print(f"\n📊 КАТЕГОРИИ (всего {len(category_results)}):")
        print(f"  Balanced Accuracy:")
        print(f"    Средневзвешенная: {ba_weighted_cat:.4f}")
        print(f"    Простое среднее:  {ba_simple_cat:.4f}")
        print(f"    Медиана:          {ba_median_cat:.4f}")
        print(f"  F1-score:")
        print(f"    Средневзвешенный: {f1_weighted_cat:.4f}")
        print(f"    Простое среднее:  {f1_simple_cat:.4f}")
        print(f"    Медиана:          {f1_median_cat:.4f}")

        # Распределение размеров категорий
        cat_sizes = [r['size'] for r in category_results.values()]
        print(f"\n  Распределение размеров категорий:")
        print(f"    Минимум:    {min(cat_sizes)}")
        print(f"    Максимум:   {max(cat_sizes)}")
        print(f"    Медиана:    {np.median(cat_sizes):.0f}")
        print(f"    75-й перц.: {np.percentile(cat_sizes, 75):.0f}")

    # Средневзвешенные метрики по кластерам
    if cluster_results:
        # С кластерными порогами
        ba_weighted_cluster, ba_simple_cluster, ba_median_cluster = calculate_weighted_metrics(
            cluster_results, 'ba_cluster_thresh'
        )
        f1_weighted_cluster, f1_simple_cluster, f1_median_cluster = calculate_weighted_metrics(
            cluster_results, 'f1_cluster_thresh'
        )

        # С оптимальными порогами
        ba_weighted_optimal, ba_simple_optimal, ba_median_optimal = calculate_weighted_metrics(
            cluster_results, 'ba_optimal_thresh'
        )
        f1_weighted_optimal, f1_simple_optimal, f1_median_optimal = calculate_weighted_metrics(
            cluster_results, 'f1_optimal_thresh'
        )

        print(f"\n📊 КЛАСТЕРЫ (всего {len(cluster_results)}):")
        print(f"\n  С КЛАСТЕРНЫМИ ПОРОГАМИ:")
        print(f"    Balanced Accuracy:")
        print(f"      Средневзвешенная: {ba_weighted_cluster:.4f}")
        print(f"      Простое среднее:  {ba_simple_cluster:.4f}")
        print(f"      Медиана:          {ba_median_cluster:.4f}")
        print(f"    F1-score:")
        print(f"      Средневзвешенный: {f1_weighted_cluster:.4f}")
        print(f"      Простое среднее:  {f1_simple_cluster:.4f}")
        print(f"      Медиана:          {f1_median_cluster:.4f}")

        print(f"\n  С ОПТИМАЛЬНЫМИ ПОРОГАМИ:")
        print(f"    Balanced Accuracy:")
        print(f"      Средневзвешенная: {ba_weighted_optimal:.4f}")
        print(f"      Простое среднее:  {ba_simple_optimal:.4f}")
        print(f"      Медиана:          {ba_median_optimal:.4f}")
        print(f"    F1-score:")
        print(f"      Средневзвешенный: {f1_weighted_optimal:.4f}")
        print(f"      Простое среднее:  {f1_simple_optimal:.4f}")
        print(f"      Медиана:          {f1_median_optimal:.4f}")

        # Улучшение
        ba_improvement_weighted = ba_weighted_optimal - ba_weighted_cluster
        ba_improvement_simple = ba_simple_optimal - ba_simple_cluster

        print(f"\n  УЛУЧШЕНИЕ (оптимальные vs кластерные пороги):")
        print(f"    Balanced Accuracy:")
        print(f"      Средневзвешенная: {ba_improvement_weighted:+.4f}")
        print(f"      Простое среднее:  {ba_improvement_simple:+.4f}")

        # Распределение размеров кластеров
        cluster_sizes = [r['size'] for r in cluster_results.values()]
        print(f"\n  Распределение размеров кластеров:")
        print(f"    Минимум:    {min(cluster_sizes)}")
        print(f"    Максимум:   {max(cluster_sizes)}")
        print(f"    Медиана:    {np.median(cluster_sizes):.0f}")
        print(f"    75-й перц.: {np.percentile(cluster_sizes, 75):.0f}")

        # Корреляция размера с качеством
        sizes = np.array(cluster_sizes)
        ba_cluster_vals = np.array([r['ba_cluster_thresh'] for r in cluster_results.values()])

        if len(sizes) > 1:
            correlation = np.corrcoef(sizes, ba_cluster_vals)[0, 1]
            print(f"    Корреляция размер-качество: {correlation:.3f}")

    # ==================== СРАВНЕНИЕ КАТЕГОРИЙ VS КЛАСТЕРОВ ====================
    if category_results and cluster_results:
        print(f'\n{"=" * 60}')
        print("СРАВНЕНИЕ КАТЕГОРИЙ И КЛАСТЕРОВ")
        print(f'{"=" * 60}')

        # Сравнение средневзвешенных BA
        print(f"\n⚖️  СРАВНЕНИЕ ПО Balanced Accuracy:")
        print(f"  Категории (средневзвешенная):     {ba_weighted_cat:.4f}")
        print(f"  Кластеры (с кластерными порогами): {ba_weighted_cluster:.4f}")
        print(f"  Кластеры (с оптимальными порогами): {ba_weighted_optimal:.4f}")

        improvement_vs_cat_cluster = ba_weighted_cluster - ba_weighted_cat
        improvement_vs_cat_optimal = ba_weighted_optimal - ba_weighted_cat

        print(f"\n📈 УЛУЧШЕНИЕ ОТНОСИТЕЛЬНО КАТЕГОРИЙ:")
        print(f"  С кластерными порогами:   {improvement_vs_cat_cluster:+.4f}")
        print(f"  С оптимальными порогами:  {improvement_vs_cat_optimal:+.4f}")

        # Анализ распределения размеров
        print(f"\n📊 АНАЛИЗ РАЗМЕРОВ ГРУПП:")
        print(f"  Категории: {len(category_results)} групп")
        print(f"  Кластеры:  {len(cluster_results)} групп")

        # Процент покрытия данных
        cat_coverage = sum(r['size'] for r in category_results.values()) / len(feat) * 100
        cluster_coverage = sum(r['size'] for r in cluster_results.values()) / len(feat) * 100

        print(f"\n📈 ПОКРЫТИЕ ДАННЫХ:")
        print(f"  Категории покрывают: {cat_coverage:.1f}% данных")
        print(f"  Кластеры покрывают:  {cluster_coverage:.1f}% данных")

    # ==================== ДЕТАЛЬНЫЙ АНАЛИЗ КРУПНЫХ ГРУПП ====================
    if cluster_results:
        print(f'\n{"=" * 60}')
        print("АНАЛИЗ КРУПНЕЙШИХ КЛАСТЕРОВ (top-10 по размеру)")
        print(f'{"=" * 60}')

        # Сортируем кластеры по размеру
        sorted_clusters = sorted(cluster_results.items(),
                                 key=lambda x: x[1]['size'],
                                 reverse=True)[:10]

        print(f"\n{'Кластер':<30} {'Размер':>8} {'Вес,%':>7} {'BA_класт':>8} {'BA_оптим':>8} {'ΔBA':>6}")
        print("-" * 75)

        total_size_top10 = 0
        total_weight_top10 = 0

        for cluster_name, metrics in sorted_clusters:
            size = metrics['size']
            weight_pct = metrics['weight'] * 100
            ba_cluster = metrics['ba_cluster_thresh']
            ba_optimal = metrics['ba_optimal_thresh']
            delta_ba = ba_optimal - ba_cluster

            total_size_top10 += size
            total_weight_top10 += metrics['weight']

            print(f"{cluster_name:<30} {size:>8} {weight_pct:>6.1f}% "
                  f"{ba_cluster:>8.3f} {ba_optimal:>8.3f} {delta_ba:>+6.3f}")

        print("-" * 75)
        print(f"Итого топ-10: {total_size_top10} объектов "
              f"({total_weight_top10 * 100:.1f}% данных)")

        # Средневзвешенная BA для топ-10
        ba_weighted_top10_cluster = sum(
            m['ba_cluster_thresh'] * m['weight'] / total_weight_top10
            for _, m in sorted_clusters
        )
        ba_weighted_top10_optimal = sum(
            m['ba_optimal_thresh'] * m['weight'] / total_weight_top10
            for _, m in sorted_clusters
        )

        print(f"Средневзвешенная BA топ-10:")
        print(f"  С кластерными порогами:  {ba_weighted_top10_cluster:.4f}")
        print(f"  С оптимальными порогами: {ba_weighted_top10_optimal:.4f}")

    return {
        'category_results': category_results,
        'cluster_results': cluster_results,
        'ba_weighted_cat': ba_weighted_cat if category_results else None,
        'ba_weighted_cluster': ba_weighted_cluster if cluster_results else None,
        'ba_weighted_optimal': ba_weighted_optimal if cluster_results else None,
        'improvement_vs_cat_cluster': improvement_vs_cat_cluster if (category_results and cluster_results) else None,
        'improvement_vs_cat_optimal': improvement_vs_cat_optimal if (category_results and cluster_results) else None
    }



def test_model(model, X_train, y_train, test_data_list, cat_features):
    print(cat_features)
    # Создаем и обучаем оптимизатор на трейне
    optimizer = LeafClusterThresholdOptimizer(
        model=model,
        n_trees=2,
        min_cluster_size=100
    )

    feature_optimizer = FeatureClusterThresholdOptimizer(
        n_clusters=30,  # Количество кластеров
        min_cluster_size=50,  # Минимальный размер кластера
        use_pca=True,  # Использовать PCA
        pca_components=20,  # Количество компонент PCA
        random_state=42
    )

    mixed_optimizer = MixedDataClusterOptimizer(
        n_clusters=20,
        min_cluster_size=50,
        categorical_distance='jaccard',  # Для one-hot кодирования
        metric='balanced_accuracy',
        cat_features=cat_features,
        random_state=42
    )


    # Обучаем на трейне
    y_pred_proba_train = model.predict_proba(X_train)[:, 1]
    #optimizer.fit(X_train, y_train, y_pred_proba_train)
    mixed_optimizer.fit(X_train, y_train, y_pred_proba_train)

    # Тестируем с кластерами (средневзвешенные метрики)
    results = test_with_clusters_weighted_mixed(
        _model=model,
        _test_data=test_data_list,
        _cat_features=cat_features,
        feature_optimizer=mixed_optimizer,
        _inference=False
    )

    # Анализ результатов
    if results['improvement_vs_cat_optimal']:
        print(f"\n{'=' * 60}")
        print("ИТОГОВОЕ ЗАКЛЮЧЕНИЕ")
        print(f"{'=' * 60}")

        if results['improvement_vs_cat_optimal'] > 0.01:
            print("✅ Кластерный подход СУЩЕСТВЕННО улучшает качество!")
        elif results['improvement_vs_cat_optimal'] > 0:
            print("✅ Кластерный подход НЕМНОГО улучшает качество")
        elif results['improvement_vs_cat_optimal'] > -0.01:
            print("⚠️  Кластерный подход НЕ УХУДШАЕТ качество")
        else:
            print("❌ Кластерный подход УХУДШАЕТ качество")

        print(
            f"\nРекомендация: {'ИСПОЛЬЗОВАТЬ' if results['improvement_vs_cat_optimal'] > 0 else 'НЕ использовать'} кластерные пороги")