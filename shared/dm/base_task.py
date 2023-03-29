from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from utils.enums import TaskMode
from utils.report import TaskReport
from utils.dataset import Dataset


class BaseTask(ABC):
    def __init__(self, config: dict, entity_embeddings: pd.DataFrame, dataset: Dataset, report: TaskReport):
        self.config = config
        self.entity_embeddings = entity_embeddings.apply(lambda x: x / np.linalg.norm(x, ord=1), axis=1)  # unit vectors
        self.dataset = dataset
        self.report = report

    @classmethod
    @abstractmethod
    def get_mode(cls) -> TaskMode:
        pass

    @abstractmethod
    def run(self):
        pass
