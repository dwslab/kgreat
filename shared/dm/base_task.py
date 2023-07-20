from abc import ABC, abstractmethod
import pandas as pd
from utils.logger import get_logger
from utils.enums import TaskType
from utils.report import TaskReport
from utils.dataset import Dataset
from utils.io import get_embedding_models, load_entity_embeddings


class BaseTask(ABC):
    def __init__(self, kg_config: dict, task_config: dict, dataset: Dataset, report: TaskReport):
        self.kg_config = kg_config
        self.task_config = task_config
        self.dataset = dataset
        self.report = report
        self.embedding_models = get_embedding_models()

    @classmethod
    @abstractmethod
    def get_type(cls) -> TaskType:
        pass

    @abstractmethod
    def run(self):
        pass

    @staticmethod
    def load_entity_embeddings(embedding_type: str, load_dataset_entities_only: bool) -> pd.DataFrame:
        get_logger().info(f'Loading entity embeddings of type {embedding_type}')
        return load_entity_embeddings(embedding_type, load_dataset_entities_only)
