from typing import List, Tuple, Type
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from utils.logger import get_logger
from utils.enums import TaskType, EntityEvalMode
from utils.dataset import TsvDataset
from base_task import BaseTask


class ClassificationTask(BaseTask):
    dataset: TsvDataset

    @classmethod
    def get_type(cls) -> TaskType:
        return TaskType.CLASSIFICATION

    METRICS = ['accuracy']
    N_SPLITS = 10

    @staticmethod
    def _get_estimators() -> List[Tuple[Type[BaseEstimator], dict]]:
        return [
            (GaussianNB, {}),
            (KNeighborsClassifier, {'n_neighbors': 1}),
            (KNeighborsClassifier, {'n_neighbors': 3}),
            (KNeighborsClassifier, {'n_neighbors': 5}),
            (SVC, {'kernel': 'rbf', 'C': 1}),
            (SVC, {'kernel': 'rbf', 'C': 0.1}),
            (SVC, {'kernel': 'linear', 'C': 1}),
            (SVC, {'kernel': 'linear', 'C': 0.1}),
        ]

    def run(self):
        entity_labels = self.dataset.get_entity_labels(mapped=True)
        fraction_of_known_entities = len(entity_labels) / (len(entity_labels) + len(self.dataset.get_entity_labels(mapped=False)))
        label_freq = entity_labels.value_counts()
        valid_labels = set(label_freq[label_freq >= self.N_SPLITS].index)
        entity_labels = entity_labels[entity_labels.isin(valid_labels)]
        if entity_labels.nunique() <= 1:
            get_logger().info(f'Skipping classification because there is at most one valid class.')
            return
        for embedding_type in self.embedding_models:
            entity_features = self.load_entity_embeddings(embedding_type, True).loc[entity_labels.index, :]
            for est, params in self._get_estimators():
                get_logger().debug(f'Evaluating classifier {est.__name__} ({params}) for embedding type {embedding_type}')
                model = est(**params)
                results = cross_validate(model, entity_features.values, entity_labels.values, scoring=self.METRICS, cv=self.N_SPLITS, n_jobs=self.N_SPLITS)
                for metric in self.METRICS:
                    known_entity_score = float(np.mean(results[f'test_{metric}']))
                    self.report.add_result(EntityEvalMode.KNOWN_ENTITIES, est.__name__, params, embedding_type, metric, known_entity_score)
                    all_entity_score = known_entity_score * fraction_of_known_entities
                    self.report.add_result(EntityEvalMode.ALL_ENTITIES, est.__name__, params, embedding_type, metric, all_entity_score)
