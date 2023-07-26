import datetime
import os
from typing import Type
from utils.enums import TaskType
from utils.io import load_dataset_config, load_kg_config, load_entity_mapping
from utils.logger import init_logger, get_logger
from utils.report import TaskReport
from utils.dataset import load_dataset
from base_task import BaseTask


class TaskManager:
    def __init__(self, task_id: str, dataset_config: dict, kg_config: dict):
        self.task_id = task_id
        self.dataset_config = dataset_config
        self.kg_config = kg_config
        self.task_config = self.kg_config['task'][task_id]

    def run_task(self):
        start_time = datetime.datetime.now()
        get_logger().info(f'Starting to run task {self.task_id}')
        # prepare mapping & dataset
        get_logger().info('Loading entity mapping')
        entity_mapping = load_entity_mapping()
        get_logger().info('Loading dataset')
        dataset = load_dataset(self.dataset_config, self.kg_config, entity_mapping)
        # prepare and run task
        get_logger().info('Running task')
        Task = self._get_task_class()
        report = TaskReport(self.task_id, Task.get_type(), dataset)
        task = Task(self.kg_config, self.task_config, dataset, report)
        task.run()
        report.store(self.kg_config['run_id'])
        runtime_in_seconds = (datetime.datetime.now() - start_time).seconds
        get_logger().info(f'Finished task after {runtime_in_seconds} seconds')

    def _get_task_class(self) -> Type[BaseTask]:
        task_type = TaskType(self.task_config['type'])
        # import modules only on demand to keep dependencies separated
        if task_type == TaskType.CLASSIFICATION:
            from classification_task import ClassificationTask
            return ClassificationTask
        if task_type == TaskType.REGRESSION:
            from regression_task import RegressionTask
            return RegressionTask
        if task_type == TaskType.CLUSTERING:
            from clustering_task import ClusteringTask
            return ClusteringTask
        if task_type == TaskType.DOCUMENT_SIMILARITY:
            from documentsimilarity_task import DocumentSimilarityTask
            return DocumentSimilarityTask
        if task_type == TaskType.ENTITY_RELATEDNESS:
            from entityrelatedness_task import EntityRelatednessTask
            return EntityRelatednessTask
        if task_type == TaskType.SEMANTIC_ANALOGIES:
            from semanticanalogies_task import SemanticAnalogiesTask
            return SemanticAnalogiesTask
        if task_type == TaskType.RECOMMENDATION:
            from recommendation_task import RecommendationTask
            return RecommendationTask
        raise NotImplementedError(f'No task implemented for type "{task_type.value}"')


if __name__ == "__main__":
    task_id = os.environ['KGREAT_STEP']
    dataset_config = load_dataset_config()
    kg_config = load_kg_config()
    init_logger(kg_config['log_level'], kg_config['run_id'], task_id)

    tm = TaskManager(task_id, dataset_config, kg_config)
    tm.run_task()
