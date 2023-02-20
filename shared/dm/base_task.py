from abc import ABC, abstractmethod
import pandas as pd
from utils.enums import TaskMode
from utils.report import TaskReport


class BaseTask(ABC):
    def __init__(self, config: dict, entity_embeddings: pd.DataFrame, dataset, report: TaskReport):
        self.config = config
        self.entity_embeddings = entity_embeddings
        self.dataset = dataset
        self.report = report

    @classmethod
    @abstractmethod
    def get_mode(cls) -> TaskMode:
        pass

    @abstractmethod
    def run(self):
        pass
