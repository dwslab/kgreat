from typing import List, Tuple, Type
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_validate
from utils.logging import get_logger
from utils.enums import TaskMode, EntityMode
from utils.dataset import TsvDataset
from base_task import BaseTask


# TODO: implement score for EntityMode.ALL_ENTITIES
class RegressionTask(BaseTask):
    dataset: TsvDataset

    @classmethod
    def get_mode(cls) -> TaskMode:
        return TaskMode.REGRESSION

    METRICS = ['neg_root_mean_squared_error']
    N_SPLITS = 10

    @staticmethod
    def _get_estimators() -> List[Tuple[Type[BaseEstimator], dict]]:
        return [
            (LinearRegression, {}),
            (KNeighborsRegressor, {'n_neighbors': 1}),
            (KNeighborsRegressor, {'n_neighbors': 3}),
            (KNeighborsRegressor, {'n_neighbors': 5}),
            (DecisionTreeRegressor, {}),
            (DecisionTreeRegressor, {'max_depth': 5}),
            (DecisionTreeRegressor, {'max_depth': 10}),
        ]

    def run(self):
        entity_labels = self.dataset.get_entity_labels()
        if len(entity_labels) < self.N_SPLITS:
            return  # skip task if we have too few samples
        for embedding_type in self.embedding_models:
            entity_features = self.load_entity_embeddings(embedding_type).loc[entity_labels.index, :]
            for est, params in self._get_estimators():
                get_logger().debug(f'Evaluating classifier {est.__name__} ({params}) for embedding type {embedding_type}')
                model = est(**params)
                results = cross_validate(model, entity_features, entity_labels, scoring=self.METRICS, cv=self.N_SPLITS, n_jobs=self.N_SPLITS)
                for metric in self.METRICS:
                    score = float(np.mean(results[f'test_{metric}']))
                    self.report.add_result(EntityMode.KNOWN_ENTITIES, est.__name__, params, embedding_type, metric, score)
