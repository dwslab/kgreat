from typing import List, Tuple, Type, Callable, Dict
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.base import BaseEstimator
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from utils.logging import get_logger
from utils.enums import TaskMode, EntityMode
from utils.dataset import TsvDataset
from base_task import BaseTask


class ClusteringTask(BaseTask):
    dataset: TsvDataset

    @classmethod
    def get_mode(cls) -> TaskMode:
        return TaskMode.CLUSTERING

    @staticmethod
    def _get_estimators(n_clusters: int) -> List[Tuple[Type[BaseEstimator], dict]]:
        return [
            (DBSCAN, {}),
            (KMeans, {'n_clusters': n_clusters}),
            (AgglomerativeClustering, {'n_clusters': n_clusters, 'linkage': 'ward'}),
            (AgglomerativeClustering, {'n_clusters': n_clusters, 'linkage': 'average'}),
        ]

    @staticmethod
    def _get_metrics() -> Dict[str, Callable]:
        return {
            'ARI': metrics.adjusted_rand_score,
            'NMI': metrics.normalized_mutual_info_score,
            'accuracy': ClusteringTask._compute_clustering_accuracy
        }

    @staticmethod
    def _compute_clustering_accuracy(y_true, y_pred) -> float:
        y_true = np.array(y_true, np.int64)
        y_pred = np.array(y_pred, np.int64)
        n_clusters = max(y_true.max(), y_pred.max()) + 1
        weights = np.zeros((n_clusters, n_clusters), np.int64)
        for pred_val, true_val in zip(y_pred, y_true):
            weights[pred_val, true_val] += 1
        row_indices, col_indices = linear_sum_assignment(weights.max() - weights)
        return sum([weights[i, j] for i, j in zip(row_indices, col_indices)]) / y_pred.size

    def run(self):
        entity_labels = self.dataset.get_entity_labels(mapped=True)
        unmapped_labels = self.dataset.get_entity_labels(mapped=False)
        n_clusters = entity_labels.nunique()
        if n_clusters <= 1:
            get_logger().debug(f'Skipping clustering task due to low number of clusters ({n_clusters})')
            return
        for embedding_type in self.embedding_models:
            entity_features = self.load_entity_embeddings(embedding_type).loc[entity_labels.index, :]
            for est, params in self._get_estimators(n_clusters):
                get_logger().debug(f'Evaluating clustering {est.__name__} ({params}) for embedding type {embedding_type}')
                model = est(**params).fit(entity_features)
                for entity_mode in [EntityMode.KNOWN_ENTITIES, EntityMode.ALL_ENTITIES]:
                    if entity_mode == EntityMode.KNOWN_ENTITIES or len(unmapped_labels) == 0:
                        predictions = model.labels_
                        true_labels = entity_labels.values
                    else:
                        predictions = list(model.labels_) + [-1] * len(unmapped_labels)
                        true_labels = list(entity_labels.values) + unmapped_labels
                    # assign entities without cluster to a unique cluster each
                    cluster_idx = n_clusters
                    for idx, cluster_id in enumerate(predictions):
                        if cluster_id == -1:
                            predictions[idx] = cluster_idx
                            cluster_idx += 1
                    # compute and report metrics
                    for metric, metric_scorer in self._get_metrics().items():
                        score = metric_scorer(true_labels, predictions)
                        self.report.add_result(entity_mode, est.__name__, params, embedding_type, metric, score)
