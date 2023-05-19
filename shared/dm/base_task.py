from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from utils.logging import get_logger
from utils.enums import TaskMode
from utils.report import TaskReport
from utils.dataset import Dataset
from utils.io import load_entity_embeddings


class BaseTask(ABC):
    def __init__(self, kg_config: dict, task_config: dict, dataset: Dataset, report: TaskReport):
        self.kg_config = kg_config
        self.embedding_models = self.kg_config['preprocessing']['embeddings']['models']
        self.task_config = task_config
        self.dataset = dataset
        self.report = report

    @classmethod
    @abstractmethod
    def get_mode(cls) -> TaskMode:
        pass

    @abstractmethod
    def run(self):
        pass

    @staticmethod
    def load_entity_embeddings(embedding_type: str) -> pd.DataFrame:
        get_logger().info(f'Loading entity embeddings of type {embedding_type}')
        embeddings = load_entity_embeddings(embedding_type)
        return embeddings.apply(lambda x: x / np.linalg.norm(x, ord=1), axis=1)  # normalize to unit vectors
