from typing import List, Tuple, Type
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from utils.enums import TaskMode
from base_task import BaseTask


class ClassificationTask(BaseTask):
    @classmethod
    def get_mode(cls) -> TaskMode:
        return TaskMode.CLASSIFICATION

    METRICS = ['accuracy']
    N_SPLITS = 10

    @staticmethod
    def _get_classifiers() -> List[Tuple[Type[BaseEstimator], dict]]:
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
        entity_labels = self.dataset.get_entity_labels()
        if len(entity_labels) < self.N_SPLITS:
            return  # skip task if we have too few samples
        entity_features = self.entity_embeddings.loc[entity_labels.index, :]
        for clf, params in self._get_classifiers():
            model = clf(**params)
            results = cross_validate(model, entity_features, entity_labels, scoring=self.METRICS, cv=self.N_SPLITS)
            for metric in self.METRICS:
                self.report.add_result(clf.__name__, params, metric, np.mean(results[f'test_{metric}']))
