from typing import List, Tuple, Type
import numpy as np
import math
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error
from utils.logging import get_logger
from utils.enums import TaskMode, EntityMode
from utils.dataset import TsvDataset
from base_task import BaseTask


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
        entity_labels = self.dataset.get_entity_labels(mapped=True)
        if len(entity_labels) < self.N_SPLITS:
            return  # skip task if we have too few samples
        # estimate error of unmapped entities by assuming a mean prediction for them -> then compute negative RMSE
        unmapped_labels = self.dataset.get_entity_labels(mapped=False)
        if len(unmapped_labels):
            unmapped_label_error = math.sqrt(mean_squared_error([float(entity_labels.mean())] * len(unmapped_labels), unmapped_labels)) * -1
        else:
            unmapped_label_error = 0
        # train models and evaluate
        for embedding_type in self.embedding_models:
            entity_features = self.load_entity_embeddings(embedding_type).loc[entity_labels.index, :]
            for est, params in self._get_estimators():
                get_logger().debug(f'Evaluating classifier {est.__name__} ({params}) for embedding type {embedding_type}')
                model = est(**params)
                results = cross_validate(model, entity_features, entity_labels, scoring=self.METRICS, cv=self.N_SPLITS, n_jobs=self.N_SPLITS)
                for metric in self.METRICS:
                    score = float(np.mean(results[f'test_{metric}']))
                    self.report.add_result(EntityMode.KNOWN_ENTITIES, est.__name__, params, embedding_type, metric, score)
                    self.report.add_result(EntityMode.ALL_ENTITIES, est.__name__, params, embedding_type, metric, score + unmapped_label_error)
