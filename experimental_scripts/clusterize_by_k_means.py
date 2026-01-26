import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances


class MixedDataClusterOptimizer:
    """
    Кластеризация для смешанных данных (числовые + категориальные).
    """

    def __init__(self, n_clusters=20, min_cluster_size=50,
                 categorical_distance='jaccard',
                 metric='balanced_accuracy',
                 cat_features=[],
                 random_state=42):
        """
        :param categorical_distance: метрика для категориальных данных
                                     ('jaccard', 'hamming', 'dice')
        """
        self.n_clusters = n_clusters
        self.min_cluster_size = min_cluster_size
        self.categorical_distance = categorical_distance
        self.metric = metric
        self.random_state = random_state
        self.cat_feat = cat_features

        self.cluster_info = {}
        self.is_fitted = False
        self.kmeans_model = None  # Добавлено!
        self.pca_model = None  # Добавлено!
        self.numeric_scaler = None  # Добавлено!
        self.scaler = None  # Для совместимости с визуализацией
        self.use_pca = True  # Добавлено!

    def _split_features(self, X):
        """Разделяет признаки на числовые и категориальные."""
        numeric_cols = []
        categorical_cols = []

        if hasattr(X, 'columns'):
            for col in X.columns:
                if col in self.cat_feat:
                    categorical_cols.append(col)
                else:
                    # Исключаем целевую переменную и служебные колонки
                    if 'log_seniority' in col or 'income_short' in col or 'age' in col:  #col != 'status' and col != 'code':
                        numeric_cols.append(col)
        else:
            # Если нет имен колонок, предполагаем что все признаки числовые
            numeric_cols = list(range(X.shape[1]))
            categorical_cols = []

        return numeric_cols, categorical_cols

    def _prepare_features_for_clustering(self, X_numeric, X_categorical, fit=False):
        """
        Подготовка признаков для кластеризации.
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        print("Подготовка признаков для кластеризации...")

        # 1. Масштабируем числовые
        if X_numeric.shape[1] > 0:
            if fit:
                self.numeric_scaler = StandardScaler()
                X_num_scaled = self.numeric_scaler.fit_transform(X_numeric)
            else:
                X_num_scaled = self.numeric_scaler.transform(X_numeric)
        else:
            X_num_scaled = X_numeric

        # 2. Для категориальных - используем агрегацию
        if X_categorical.shape[1] > 0:
            # Агрегируем категориальные признаки
            cat_mean = X_categorical.mean(axis=1, keepdims=True)
            cat_sum = X_categorical.sum(axis=1, keepdims=True)

            # Вычисляем энтропию распределения
            if fit:
                p = X_categorical.mean(axis=0)
                p = p[p > 0]
                self.cat_entropy_value = -np.sum(p * np.log2(p)) if len(p) > 0 else 0

            cat_entropy = np.full((X_categorical.shape[0], 1), self.cat_entropy_value)

            X_cat_aggregated = np.hstack([cat_mean, cat_sum, cat_entropy])
            print(f"Агрегировали {X_categorical.shape[1]} категориальных в 3 признака")
        else:
            X_cat_aggregated = np.array([]).reshape(X_numeric.shape[0], 0)

        # 3. Объединяем
        if X_num_scaled.shape[1] > 0 and X_cat_aggregated.shape[1] > 0:
            X_combined = np.hstack([X_num_scaled, X_cat_aggregated])
        elif X_num_scaled.shape[1] > 0:
            X_combined = X_num_scaled
        else:
            X_combined = X_cat_aggregated

        # 4. PCA для дальнейшего сокращения
        n_components = min(20, X_combined.shape[1], X_combined.shape[0] - 1)
        if n_components > 0:
            if fit:
                self.pca_model = PCA(n_components=n_components, random_state=self.random_state)
                X_final = self.pca_model.fit_transform(X_combined)
                print(f"PCA: объясненная дисперсия = {self.pca_model.explained_variance_ratio_.sum():.3f}")
            else:
                X_final = self.pca_model.transform(X_combined)
        else:
            X_final = X_combined

        print(f"Итоговые признаки для кластеризации: {X_final.shape[1]}")
        return X_final

    def fit(self, X, y, y_pred_proba):
        """Обучение кластеризации на смешанных данных."""
        from sklearn.cluster import KMeans

        print(f"Кластеризация смешанных данных ({len(X)} объектов)...")

        # Разделяем признаки
        numeric_cols, categorical_cols = self._split_features(X)

        if hasattr(X, 'columns'):
            X_numeric = X[numeric_cols].values if numeric_cols else np.array([]).reshape(len(X), 0)
            X_categorical = X[categorical_cols].values.astype(float) if categorical_cols else np.array([]).reshape(
                len(X), 0)
        else:
            X_numeric = X
            X_categorical = np.array([]).reshape(len(X), 0)

        print(f"  Числовые признаки: {len(numeric_cols)}")
        print(f"  Категориальные признаки: {len(categorical_cols)}")

        # Подготавливаем признаки
        X_processed = self._prepare_features_for_clustering(X_numeric, X_categorical, fit=True)

        # Используем K-means
        self.kmeans_model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        cluster_labels = self.kmeans_model.fit_predict(X_processed)

        print(f"Кластеризация завершена. Получено {self.n_clusters} кластеров")

        # 3. Анализ размеров кластеров
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        print("\nРаспределение по кластерам:")
        for label, count in zip(unique_labels, counts):
            print(f"  Кластер {label}: {count} объектов ({count / len(X) * 100:.1f}%)")

        # 4. Объединяем мелкие кластеры
        self.cluster_info = {}
        self.cluster_labels_ = np.full(len(X), "Unknown", dtype=object)

        # Создаем словарь для хранения индексов по кластерам
        cluster_to_indices = {}
        for idx, label in enumerate(cluster_labels):
            if label not in cluster_to_indices:
                cluster_to_indices[label] = []
            cluster_to_indices[label].append(idx)

        # Обрабатываем каждый кластер
        valid_cluster_counter = 0

        for cluster_id, indices in cluster_to_indices.items():
            cluster_size = len(indices)

            if cluster_size >= self.min_cluster_size:
                cluster_name = f"Cluster_{valid_cluster_counter}"
                valid_cluster_counter += 1

                # Подбираем порог для этого кластера
                y_true_cluster = y.iloc[indices] if hasattr(y, 'iloc') else y[indices]
                y_proba_cluster = y_pred_proba[indices]

                # Подбор порога, максимизирующего Balanced Accuracy
                best_threshold, best_ba = self._find_optimal_threshold(
                    y_true_cluster, y_proba_cluster
                )

                # Сохраняем информацию о кластере
                self.cluster_info[cluster_name] = {
                    'threshold': best_threshold,
                    'balanced_acc': best_ba,
                    'size': cluster_size,
                    'indices': indices,
                    'mean_proba': np.mean(y_proba_cluster),
                    'std_proba': np.std(y_proba_cluster),
                    'pos_rate': np.mean(y_true_cluster),
                    'original_kmeans_label': cluster_id
                }

                # Сохраняем метки
                self.cluster_labels_[indices] = cluster_name
            else:
                # Мелкие кластеры пока не обрабатываем
                pass

        # 5. Создаем кластер "Other" для мелких групп и неклассифицированных
        other_indices = np.where(self.cluster_labels_ == "Unknown")[0].tolist()

        if other_indices:
            cluster_name = "Cluster_Other"

            # Для кластера "Other" используем порог 0.5 или средний порог
            avg_threshold = np.mean([info['threshold'] for info in self.cluster_info.values()]) \
                if self.cluster_info else 0.5

            self.cluster_info[cluster_name] = {
                'threshold': avg_threshold,
                'balanced_acc': 0.0,
                'size': len(other_indices),
                'indices': other_indices,
                'mean_proba': np.mean(y_pred_proba[other_indices]),
                'std_proba': np.std(y_pred_proba[other_indices]),
                'pos_rate': np.mean(y.iloc[other_indices] if hasattr(y, 'iloc') else y[other_indices]),
                'original_kmeans_label': -1
            }

            self.cluster_labels_[other_indices] = cluster_name

        print(f"\nИтоговое количество значимых кластеров: {len(self.cluster_info)}")
        if "Cluster_Other" in self.cluster_info:
            print(f"Кластер 'Other' содержит {self.cluster_info['Cluster_Other']['size']} объектов")

        # 6. Вывод информации о кластерах
        print("\nПодробная информация о кластерах:")
        print("-" * 70)
        for cluster_name, info in self.cluster_info.items():
            print(f"{cluster_name:<15} | size: {info['size']:>5} | "
                  f"threshold: {info['threshold']:.3f} | BA: {info['balanced_acc']:.3f} | "
                  f"pos_rate: {info['pos_rate']:.3f}")

        # ВАЖНО: Устанавливаем флаг, что модель обучена!
        self.is_fitted = True
        self.scaler = self.numeric_scaler  # Для совместимости

        return self

    def _find_optimal_threshold(self, y_true, y_proba):
        """Находит оптимальный порог по Balanced Accuracy."""
        from sklearn.metrics import balanced_accuracy_score

        best_threshold = 0.5
        best_ba = 0.0

        # Проверяем, есть ли оба класса
        if len(np.unique(y_true)) < 2:
            return best_threshold, best_ba

        # Диапазон для поиска порога
        proba_range = np.percentile(y_proba, [10, 90])
        thresholds = np.linspace(max(0.1, proba_range[0]), min(0.9, proba_range[1]), 50)

        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            ba = balanced_accuracy_score(y_true, y_pred)

            if ba > best_ba:
                best_ba = ba
                best_threshold = threshold

        return best_threshold, best_ba

    def get_cluster_labels(self):
        """Возвращает метки кластеров для всех объектов."""
        if not self.is_fitted:
            raise ValueError("Модель не обучена. Сначала вызовите fit().")
        return self.cluster_labels_

    def get_cluster_summary(self):
        """Возвращает DataFrame с информацией о всех кластерах."""
        if not self.is_fitted:
            raise ValueError("Модель не обучена. Сначала вызовите fit().")

        summary_data = []
        for cluster_name, info in self.cluster_info.items():
            summary_data.append({
                'cluster': cluster_name,
                'size': info['size'],
                'threshold': info['threshold'],
                'balanced_acc': info['balanced_acc'],
                'mean_proba': info['mean_proba'],
                'std_proba': info['std_proba'],
                'pos_rate': info.get('pos_rate', np.nan),
                'original_label': info.get('original_kmeans_label', -1)
            })
        return pd.DataFrame(summary_data).sort_values('size', ascending=False)

    def predict_cluster(self, X_new):
        """
        Предсказание кластера для новых данных.

        :param X_new: новые данные (DataFrame или массив)
        :return: массив меток кластеров
        """
        if not self.is_fitted:
            raise ValueError("Модель не обучена. Сначала вызовите fit().")

        # Разделяем признаки
        numeric_cols, categorical_cols = self._split_features(X_new)

        if hasattr(X_new, 'columns'):
            X_numeric = X_new[numeric_cols].values if numeric_cols else np.array([]).reshape(len(X_new), 0)
            X_categorical = X_new[categorical_cols].values.astype(float) if categorical_cols else np.array([]).reshape(
                len(X_new), 0)
        else:
            X_numeric = X_new
            X_categorical = np.array([]).reshape(len(X_new), 0)

        # Подготавливаем признаки
        X_processed = self._prepare_features_for_clustering(X_numeric, X_categorical, fit=False)

        # Предсказание K-means
        kmeans_labels = self.kmeans_model.predict(X_processed)

        # Сопоставляем с нашими кластерами
        cluster_labels_new = []

        for kmeans_label in kmeans_labels:
            # Ищем кластер с таким original_kmeans_label
            found = False
            for cluster_name, info in self.cluster_info.items():
                if info.get('original_kmeans_label') == kmeans_label:
                    cluster_labels_new.append(cluster_name)
                    found = True
                    break

            # Если не нашли, попадает в "Other"
            if not found:
                cluster_labels_new.append("Cluster_Other")

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

class FeatureClusterThresholdOptimizer:
    """
    Кластеризация по признакам с помощью K-means и подбор порогов по Balanced Accuracy.
    """

    def __init__(self, n_clusters=20, min_cluster_size=50,
                 use_pca=False, pca_components=None,
                 random_state=42):
        """
        :param n_clusters: количество кластеров для K-means
        :param min_cluster_size: минимальный размер кластера для подбора порога
        :param use_pca: использовать ли PCA для уменьшения размерности
        :param pca_components: количество компонент PCA
        :param random_state: seed для воспроизводимости
        """
        self.n_clusters = n_clusters
        self.min_cluster_size = min_cluster_size
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.random_state = random_state

        self.cluster_info = {}  # {cluster_name: {threshold: float, size: int, ...}}
        self.kmeans_model = None
        self.pca_model = None
        self.scaler = None
        self.feature_names = None

    def _prepare_features(self, X):
        """Подготовка признаков для кластеризации."""
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        # Сохраняем имена признаков
        if hasattr(X, 'columns'):
            self.feature_names = X.columns.tolist()

        # Преобразуем в numpy array
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X

        # Масштабирование
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_array)

        # PCA если нужно
        if self.use_pca:
            if self.pca_components is None:
                self.pca_components = min(50, X_scaled.shape[1])

            self.pca_model = PCA(n_components=self.pca_components,
                                 random_state=self.random_state)
            X_processed = self.pca_model.fit_transform(X_scaled)
            print(f"PCA: сократили с {X_scaled.shape[1]} до {X_processed.shape[1]} признаков")
        else:
            X_processed = X_scaled

        return X_processed

    def fit(self, X, y, y_pred_proba):
        """
        Обучение кластеризации и подбор порогов для каждого кластера.

        :param X: признаки
        :param y: истинные метки
        :param y_pred_proba: предсказанные вероятности от модели
        :return: self
        """
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        print(f"Кластеризация {len(X)} объектов с помощью K-means...")

        # 1. Подготовка признаков
        X_processed = self._prepare_features(X)
        print(f"Признаки после обработки: {X_processed.shape}")

        # 2. Кластеризация K-means
        self.kmeans_model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
            max_iter=300
        )

        cluster_labels = self.kmeans_model.fit_predict(X_processed)
        print(f"Кластеризация завершена. Получено {self.n_clusters} кластеров")

        # 3. Анализ размеров кластеров
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        print("\nРаспределение по кластерам:")
        for label, count in zip(unique_labels, counts):
            print(f"  Кластер {label}: {count} объектов ({count / len(X) * 100:.1f}%)")

        # 4. Объединяем мелкие кластеры
        self.cluster_info = {}
        self.cluster_labels_ = np.full(len(X), "Unknown", dtype=object)

        # Создаем словарь для хранения индексов по кластерам
        cluster_to_indices = {}
        for idx, label in enumerate(cluster_labels):
            if label not in cluster_to_indices:
                cluster_to_indices[label] = []
            cluster_to_indices[label].append(idx)

        # Обрабатываем каждый кластер
        valid_cluster_counter = 0

        for cluster_id, indices in cluster_to_indices.items():
            cluster_size = len(indices)

            if cluster_size >= self.min_cluster_size:
                cluster_name = f"Cluster_{valid_cluster_counter}"
                valid_cluster_counter += 1

                # Подбираем порог для этого кластера
                y_true_cluster = y.iloc[indices] if hasattr(y, 'iloc') else y[indices]
                y_proba_cluster = y_pred_proba[indices]

                # Подбор порога, максимизирующего Balanced Accuracy
                best_threshold, best_ba = self._find_optimal_threshold(
                    y_true_cluster, y_proba_cluster
                )

                # Сохраняем информацию о кластере
                self.cluster_info[cluster_name] = {
                    'threshold': best_threshold,
                    'balanced_acc': best_ba,
                    'size': cluster_size,
                    'indices': indices,
                    'mean_proba': np.mean(y_proba_cluster),
                    'std_proba': np.std(y_proba_cluster),
                    'pos_rate': np.mean(y_true_cluster),
                    'original_kmeans_label': cluster_id
                }

                # Сохраняем метки
                self.cluster_labels_[indices] = cluster_name
            else:
                # Мелкие кластеры пока не обрабатываем
                pass

        # 5. Создаем кластер "Other" для мелких групп и неклассифицированных
        other_indices = np.where(self.cluster_labels_ == "Unknown")[0].tolist()

        if other_indices:
            cluster_name = "Cluster_Other"

            # Для кластера "Other" используем порог 0.5 или средний порог
            avg_threshold = np.mean([info['threshold'] for info in self.cluster_info.values()]) \
                if self.cluster_info else 0.5

            self.cluster_info[cluster_name] = {
                'threshold': avg_threshold,
                'balanced_acc': 0.0,
                'size': len(other_indices),
                'indices': other_indices,
                'mean_proba': np.mean(y_pred_proba[other_indices]),
                'std_proba': np.std(y_pred_proba[other_indices]),
                'pos_rate': np.mean(y.iloc[other_indices] if hasattr(y, 'iloc') else y[other_indices]),
                'original_kmeans_label': -1
            }

            self.cluster_labels_[other_indices] = cluster_name

        print(f"\nИтоговое количество значимых кластеров: {len(self.cluster_info)}")
        if "Cluster_Other" in self.cluster_info:
            print(f"Кластер 'Other' содержит {self.cluster_info['Cluster_Other']['size']} объектов")

        # 6. Вывод информации о кластерах
        print("\nПодробная информация о кластерах:")
        print("-" * 70)
        for cluster_name, info in self.cluster_info.items():
            print(f"{cluster_name:<15} | size: {info['size']:>5} | "
                  f"threshold: {info['threshold']:.3f} | BA: {info['balanced_acc']:.3f} | "
                  f"pos_rate: {info['pos_rate']:.3f}")

        return self

    def _find_optimal_threshold(self, y_true, y_proba):
        """Находит оптимальный порог по Balanced Accuracy."""
        from sklearn.metrics import balanced_accuracy_score

        best_threshold = 0.5
        best_ba = 0.0

        # Проверяем, есть ли оба класса
        if len(np.unique(y_true)) < 2:
            return best_threshold, best_ba

        # Диапазон для поиска порога
        proba_range = np.percentile(y_proba, [10, 90])
        thresholds = np.linspace(max(0.1, proba_range[0]), min(0.9, proba_range[1]), 50)

        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            ba = balanced_accuracy_score(y_true, y_pred)

            if ba > best_ba:
                best_ba = ba
                best_threshold = threshold

        return best_threshold, best_ba

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
                'pos_rate': info.get('pos_rate', np.nan),
                'original_label': info.get('original_kmeans_label', -1)
            })
        return pd.DataFrame(summary_data).sort_values('size', ascending=False)

    def predict_cluster(self, X_new):
        """
        Предсказание кластера для новых данных.

        :param X_new: новые данные (DataFrame или массив)
        :return: массив меток кластеров
        """
        # Подготовка признаков
        if isinstance(X_new, pd.DataFrame):
            X_new_array = X_new.values
        else:
            X_new_array = X_new

        # Масштабирование
        X_new_scaled = self.scaler.transform(X_new_array)

        # PCA если использовали
        if self.use_pca and self.pca_model is not None:
            X_new_processed = self.pca_model.transform(X_new_scaled)
        else:
            X_new_processed = X_new_scaled

        # Предсказание K-means
        kmeans_labels = self.kmeans_model.predict(X_new_processed)

        # Сопоставляем с нашими кластерами
        cluster_labels_new = []

        for kmeans_label in kmeans_labels:
            # Ищем кластер с таким original_kmeans_label
            found = False
            for cluster_name, info in self.cluster_info.items():
                if info.get('original_kmeans_label') == kmeans_label:
                    cluster_labels_new.append(cluster_name)
                    found = True
                    break

            # Если не нашли, попадает в "Other"
            if not found:
                cluster_labels_new.append("Cluster_Other")

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

    def visualize_clusters(self, X, y=None, top_n=15):
        """
        Визуализация кластеров.

        :param X: данные для визуализации
        :param y: целевая переменная (опционально)
        :param top_n: количество топ-кластеров для отображения
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Получаем кластеры
        clusters = self.get_cluster_labels()

        # Визуализация 1: Размеры кластеров
        plt.figure(figsize=(12, 5))

        cluster_counts = pd.Series(clusters).value_counts()
        top_clusters = cluster_counts.head(top_n)

        plt.subplot(1, 2, 1)
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_clusters)))
        bars = plt.bar(range(len(top_clusters)), top_clusters.values, color=colors)
        plt.xticks(range(len(top_clusters)), top_clusters.index, rotation=45, ha='right')
        plt.xlabel('Кластер')
        plt.ylabel('Количество объектов')
        plt.title(f'Топ-{top_n} кластеров по размеру')

        # Добавляем пороговые значения
        for i, (cluster_name, count) in enumerate(top_clusters.items()):
            if cluster_name in self.cluster_info:
                threshold = self.cluster_info[cluster_name]['threshold']
                plt.text(i, count + max(top_clusters.values) * 0.01,
                         f'{threshold:.2f}', ha='center', fontsize=8)

        # Визуализация 2: PCA проекция (если используется PCA)
        if self.use_pca and hasattr(self, 'pca_model'):
            plt.subplot(1, 2, 2)

            # Подготовка данных
            if isinstance(X, pd.DataFrame):
                X_array = X.values
            else:
                X_array = X

            X_scaled = self.scaler.transform(X_array)
            X_pca = self.pca_model.transform(X_scaled)

            # Берем только первые 2 компоненты
            if X_pca.shape[1] >= 2:
                # Выбираем только объекты из топ-кластеров
                top_cluster_names = set(top_clusters.index)
                mask = np.array([cluster in top_cluster_names for cluster in clusters])

                if mask.any():
                    # Создаем цветовую карту
                    cluster_to_color = {name: i / len(top_cluster_names)
                                        for i, name in enumerate(top_cluster_names)}
                    colors_2d = [cluster_to_color.get(cluster, 0.5)
                                 for cluster in clusters[mask]]

                    plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                                c=colors_2d, cmap='tab20', alpha=0.6, s=10)
                    plt.xlabel('PCA Component 1')
                    plt.ylabel('PCA Component 2')
                    plt.title('Проекция кластеров на PCA (первые 2 компоненты)')
                    plt.colorbar(label='Кластер')

        plt.tight_layout()
        plt.show()