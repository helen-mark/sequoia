import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import balanced_accuracy_score
import warnings
import pickle

warnings.filterwarnings('ignore')


class LeafClusterThresholdOptimizer:
    """
    Кластеризация по первым N деревьям CatBoost с подбором порогов по Balanced Accuracy.
    """

    def __init__(self, model, n_trees=10, min_cluster_size=50, random_state=42):
        """
        :param model: обученная модель CatBoost
        :param n_trees: количество первых деревьев для кластеризации
        :param min_cluster_size: минимальный размер кластера для подбора порога
        :param random_state: seed для воспроизводимости
        """
        self.model = model
        self.n_trees = n_trees
        self.min_cluster_size = min_cluster_size
        self.random_state = random_state
        self.cluster_info = {}  # {cluster_name: {threshold: float, size: int, ...}}
        self.cluster_encoder = {}  # {leaf_signature: cluster_name}
        self.reverse_encoder = {}  # {cluster_name: leaf_signature}

    def _get_leaf_signature(self, leaf_indices_row):
        """Создаёт уникальную строковую сигнатуру из индексов листьев первых N деревьев."""
        # Преобразуем индексы листьев первых n_trees деревьев в строку
        return '_'.join(str(int(leaf_idx)) for leaf_idx in leaf_indices_row[:self.n_trees])

    def _get_readable_cluster_name(self, leaf_signature, cluster_num):
        """Создаёт читаемое имя кластера вида Cluster_N_XXXX."""
        # Берём первые 4 элемента сигнатуры для краткости
        signature_part = '_'.join(leaf_signature.split('_')[:4])
        return f"Cluster_{cluster_num}_{signature_part}"

    def fit(self, X, y, y_pred_proba):
        """
        Обучение кластеризации и подбор порогов для каждого кластера.

        :param X: признаки
        :param y: истинные метки
        :param y_pred_proba: предсказанные вероятности от модели
        :return: self
        """
        print(f"Получение индексов листьев для {len(X)} объектов...")

        # 1. Получаем все индексы листьев
        pool = Pool(X, cat_features=self.model.get_cat_feature_indices())
        all_leaf_indices = self.model.calc_leaf_indexes(pool)  # [n_trees, n_samples]

        # Транспонируем для удобства: [n_samples, n_trees]
        leaf_indices_array = np.array(all_leaf_indices).T
        print(f"Размерность матрицы листьев: {leaf_indices_array.shape}")

        # ВАЖНО: Проверьте транспонирование!
        print(f"leaf_indices_array.shape ДО транспонирования: {leaf_indices_array.shape}")

        # Правильный вариант:
        if leaf_indices_array.shape[0] == 800:  # Если деревья в первой оси
            print("⚠️ Требуется транспонирование!")
            leaf_indices_array = leaf_indices_array.T  # Теперь (58685, 800)
            print(f"ПОСЛЕ транспонирования: {leaf_indices_array.shape}")

        # 2. Создаём сигнатуры для каждого объекта
        signatures = [self._get_leaf_signature(row) for row in leaf_indices_array]
        unique_signatures = list(set(signatures))
        print(f"Найдено уникальных сигнатур: {len(unique_signatures)}")

        # 3. Группируем объекты по сигнатурам (это и есть наши кластеры)
        signature_to_indices = {}
        for idx, sig in enumerate(signatures):
            if sig not in signature_to_indices:
                signature_to_indices[sig] = []
            signature_to_indices[sig].append(idx)

        # 4. Объединяем мелкие кластеры в "Other"
        self.cluster_info = {}
        self.cluster_encoder = {}
        self.reverse_encoder = {}

        cluster_counter = 0
        main_clusters = []
        other_indices = []

        for sig, indices in signature_to_indices.items():
            if len(indices) >= self.min_cluster_size:
                # Создаём имя для основного кластера
                cluster_name = self._get_readable_cluster_name(sig, cluster_counter)
                self.cluster_encoder[sig] = cluster_name
                self.reverse_encoder[cluster_name] = sig
                main_clusters.append((cluster_name, indices))
                cluster_counter += 1
            else:
                other_indices.extend(indices)

        # 5. Создаём кластер "Other" для мелких групп
        if other_indices:
            cluster_name = f"Cluster_Other"
            self.cluster_encoder["other"] = cluster_name
            self.reverse_encoder[cluster_name] = "other"
            main_clusters.append((cluster_name, other_indices))
            print(f"Создан кластер 'Other' размером {len(other_indices)} объектов")

        print(f"\nИтоговое количество кластеров: {len(main_clusters)}")

        # 6. Для каждого кластера подбираем оптимальный порог по Balanced Accuracy
        print("\nПодбор порогов для каждого кластера...")
        print("-" * 60)

        for cluster_name, indices in main_clusters:
            if len(indices) < 10:  # Слишком мало для подбора порога
                self.cluster_info[cluster_name] = {
                    'threshold': 0.5,
                    'balanced_acc': 0.0,
                    'size': len(indices),
                    'indices': indices,
                    'mean_proba': np.mean(y_pred_proba[indices]),
                    'std_proba': np.std(y_pred_proba[indices])
                }
                continue

            y_true_cluster = y.iloc[indices] if hasattr(y, 'iloc') else y[indices]
            y_proba_cluster = y_pred_proba[indices]

            # Подбор порога, максимизирующего Balanced Accuracy
            best_threshold = 0.5
            best_ba = 0.0

            # Диапазон для поиска порога от 10-го до 90-го перцентиля вероятностей
            proba_range = np.percentile(y_proba_cluster, [10, 90])
            thresholds = np.linspace(max(0.1, proba_range[0]), min(0.9, proba_range[1]), 50)

            for threshold in thresholds:
                y_pred_cluster = (y_proba_cluster >= threshold).astype(int)
                ba = balanced_accuracy_score(y_true_cluster, y_pred_cluster)

                if ba > best_ba:
                    best_ba = ba
                    best_threshold = threshold

            # Сохраняем информацию о кластере
            self.cluster_info[cluster_name] = {
                'threshold': best_threshold,
                'balanced_acc': best_ba,
                'size': len(indices),
                'indices': indices,
                'mean_proba': np.mean(y_proba_cluster),
                'std_proba': np.std(y_proba_cluster),
                'pos_rate': np.mean(y_true_cluster)
            }

            print(f"{cluster_name:<25} | size: {len(indices):>5} | "
                  f"threshold: {best_threshold:.3f} | BA: {best_ba:.3f} | "
                  f"mean proba: {np.mean(y_proba_cluster):.3f}")

        # 7. Создаём массив меток кластеров для всех объектов
        self.cluster_labels_ = np.full(len(X), "Unknown", dtype=object)
        for cluster_name, indices in main_clusters:
            self.cluster_labels_[indices] = cluster_name

        print("-" * 60)
        print("Кластеризация завершена.")

        return self

    def get_cluster_labels(self):
        """Возвращает метки кластеров для всех объектов."""
        return self.cluster_labels_

    def get_cluster_summary(self):
        """Возвращает DataFrame с информацией о всех кластерах."""
        summary_data = []
        for cluster_name, info in self.cluster_info.items():
            summary_data.append({
                'cluster': cluster_name,
                'size': info['size'],
                'threshold': info['threshold'],
                'balanced_acc': info['balanced_acc'],
                'mean_proba': info['mean_proba'],
                'std_proba': info['std_proba'],
                'pos_rate': info.get('pos_rate', np.nan)
            })
        return pd.DataFrame(summary_data).sort_values('size', ascending=False)

    def predict_cluster(self, X_new):
        """
        Предсказание кластера для новых данных.

        :param X_new: новые данные (DataFrame или массив)
        :return: массив меток кластеров
        """
        # Получаем индексы листьев для новых данных
        pool_new = Pool(X_new, cat_features=self.model.get_cat_feature_indices())
        leaf_indices_new = self.model.calc_leaf_indexes(pool_new)
        leaf_indices_array_new = np.array(leaf_indices_new).T

        if leaf_indices_array_new.shape[0] == self.model.get_params().get('iterations', 800):
            leaf_indices_array_new = leaf_indices_array_new.T

        # Создаём сигнатуры
        signatures_new = [self._get_leaf_signature(row) for row in leaf_indices_array_new]

        # Сопоставляем сигнатуры с кластерами
        cluster_labels_new = []
        for sig in signatures_new:
            if sig in self.cluster_encoder:
                cluster_labels_new.append(self.cluster_encoder[sig])
            else:
                # Если сигнатура новая, проверяем похожесть на существующие кластеры
                # Простой метод: находим ближайшую известную сигнатуру по расстоянию Хэмминга
                if self.reverse_encoder.get("Cluster_Other"):
                    cluster_labels_new.append("Cluster_Other")
                else:
                    # Разбиваем сигнатуру на части и пытаемся найти частичное совпадение
                    sig_parts = set(sig.split('_'))
                    best_match = "Unknown"
                    best_score = 0

                    for known_sig, cluster_name in self.cluster_encoder.items():
                        if known_sig == "other":
                            continue
                        known_parts = set(known_sig.split('_')[-self.n_trees:])  # Без префикса "Cluster_N_"
                        overlap = len(sig_parts.intersection(known_parts))
                        if overlap > best_score:
                            best_score = overlap
                            best_match = cluster_name

                    cluster_labels_new.append(best_match if best_score > 0 else "Cluster_Other")

        return np.array(cluster_labels_new)

    def predict_with_cluster_thresholds(self, X_new, y_pred_proba_new):
        """
        Предсказание с учётом кластерных порогов.

        :param X_new: новые данные
        :param y_pred_proba_new: вероятности от модели для новых данных
        :return: (final_predictions, cluster_labels)
        """
        # 1. Определяем кластер для каждого примера
        cluster_labels = self.predict_cluster(X_new)

        # 2. Применяем соответствующий порог
        final_predictions = np.zeros(len(X_new), dtype=int)

        for cluster_name in np.unique(cluster_labels):
            cluster_mask = (cluster_labels == cluster_name)
            if cluster_mask.any():
                threshold = self.cluster_info.get(cluster_name, {}).get('threshold', 0.5)
                final_predictions[cluster_mask] = (y_pred_proba_new[cluster_mask] >= threshold).astype(int)

        return final_predictions, cluster_labels

    def analyze_cluster_stability(self, X_old, X_new):
        """
        Анализирует стабильность кластеров между старыми и новыми данными.

        :param X_old: старые данные (например, 2020-2023)
        :param X_new: новые данные (2024)
        :return: DataFrame со статистикой стабильности
        """
        # Определяем кластеры для старых и новых данных
        clusters_old = self.predict_cluster(X_old)
        clusters_new = self.predict_cluster(X_new)

        # Считаем распределение по кластерам
        old_counts = pd.Series(clusters_old).value_counts(normalize=True)
        new_counts = pd.Series(clusters_new).value_counts(normalize=True)

        # Объединяем в одну таблицу
        stability_df = pd.DataFrame({
            'old_percent': old_counts,
            'new_percent': new_counts
        }).fillna(0)

        # Считаем изменение
        stability_df['change'] = stability_df['new_percent'] - stability_df['old_percent']
        stability_df['change_abs'] = abs(stability_df['change'])

        # Сортируем по величине изменения
        stability_df = stability_df.sort_values('change_abs', ascending=False)

        # Вычисляем общую стабильность (1 - total variation distance)
        total_variation = 0.5 * stability_df['change_abs'].sum()
        stability_score = 1 - total_variation

        print(f"Общий показатель стабильности кластеров: {stability_score:.3f}")
        print(f"(1.0 = идеальная стабильность, 0.0 = полное изменение)")

        return stability_df, stability_score

    def visualize_cluster_distributions(self, X, y=None, top_n=10):
        """
        Визуализация распределений по кластерам.

        :param X: данные для визуализации
        :param y: целевая переменная (опционально)
        :param top_n: количество топ-кластеров для отображения
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Получаем кластеры
        clusters = self.predict_cluster(X)
        cluster_counts = pd.Series(clusters).value_counts()

        # Визуализация 1: Размеры кластеров
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        top_clusters = cluster_counts.head(top_n)
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_clusters)))
        bars = plt.bar(range(len(top_clusters)), top_clusters.values, color=colors)
        plt.xticks(range(len(top_clusters)), top_clusters.index, rotation=45, ha='right')
        plt.xlabel('Кластер')
        plt.ylabel('Количество объектов')
        plt.title(f'Топ-{top_n} кластеров по размеру')

        # Добавляем пороговые значения на график
        for i, (cluster_name, count) in enumerate(top_clusters.items()):
            if cluster_name in self.cluster_info:
                threshold = self.cluster_info[cluster_name]['threshold']
                plt.text(i, count + max(top_clusters.values)*0.01,
                        f'{threshold:.2f}', ha='center', fontsize=8)

        # Визуализация 2: Распределение вероятностей по кластерам (если есть y_pred_proba)
        if hasattr(self, 'y_pred_proba_train'):
            plt.subplot(1, 2, 2)

            # Собираем данные для боксплота
            plot_data = []
            for cluster_name in top_clusters.index:
                if cluster_name in self.cluster_info:
                    indices = self.cluster_info[cluster_name]['indices']
                    proba_values = self.y_pred_proba_train[indices]
                    for proba in proba_values:
                        plot_data.append({'cluster': cluster_name, 'proba': proba})

            if plot_data:
                plot_df = pd.DataFrame(plot_data)
                sns.boxplot(x='cluster', y='proba', data=plot_df)
                plt.xticks(rotation=45, ha='right')
                plt.xlabel('Кластер')
                plt.ylabel('Вероятность')
                plt.title('Распределение вероятностей по кластерам')

                # Добавляем горизонтальные линии порогов
                for i, cluster_name in enumerate(top_clusters.index):
                    if cluster_name in self.cluster_info:
                        threshold = self.cluster_info[cluster_name]['threshold']
                        plt.axhline(y=threshold, xmin=i/len(top_clusters),
                                  xmax=(i+1)/len(top_clusters),
                                  color='red', alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.show()

# Пример использования
if __name__ == "__main__":
    dataset_src = 'data/trn-4'
    test_src = 'data/tst-4'
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)

    X_train = pd.read_csv('x_train.csv').drop(columns='code')
    X_test = pd.read_csv('x_val.csv').drop(columns='code')
    y_train = pd.read_csv('y_train.csv')['status']

    y_pred_proba_train = model.predict_proba(X_train)[:, 1]

    optimizer = LeafClusterThresholdOptimizer(
        model=model,
        n_trees=7,           # Используем первые 10 деревьев
        min_cluster_size=100  # Минимальный размер кластера для подбора порога
    )

    optimizer.fit(X_train, y_train, y_pred_proba_train)

    # 4. Получаем информацию о кластерах
    cluster_summary = optimizer.get_cluster_summary()
    print(cluster_summary)

    # 5. Получаем метки кластеров для трейна
    train_clusters = optimizer.get_cluster_labels()

    # 6. Применяем к новым данным
    y_pred_proba_test = model.predict_proba(X_test)[:, 1]

    print("Test data")
    final_predictions, test_clusters = optimizer.predict_with_cluster_thresholds(
        X_test,
        y_pred_proba_test
    )

    # 7. Анализ стабильности (если есть разделение на старые и новые данные)
    stability_df, score = optimizer.analyze_cluster_stability(X_train, X_test)

    # 8. Визуализация
    optimizer.visualize_cluster_distributions(X_train, top_n=15)
