import datetime
import os
import yaml
from typing import Type
from utils.enums import TaskMode
from utils.io import load_kg_config, load_entity_mapping
from utils.logging import init_logger, get_logger
from utils.report import TaskReport
from utils.dataset import load_dataset
from base_task import BaseTask


class TaskManager:
    def __init__(self, task_id: str, dataset_config: dict):
        self.task_id = task_id
        self.dataset_config = dataset_config
        self.kg_config = load_kg_config()
        self.task_config = self.kg_config['tasks'][task_id]

    def run_task(self):
        start_time = datetime.datetime.now()
        init_logger(self.kg_config['run_id'], self.task_id, self.kg_config['log_level'])
        get_logger().info(f'Starting to run task {self.task_id}')
        # prepare mapping & dataset
        get_logger().info('Loading entity mapping')
        entity_mapping = load_entity_mapping()
        get_logger().info('Loading dataset')
        dataset = load_dataset(self.dataset_config, self.kg_config, entity_mapping)
        # prepare and run task
        get_logger().info('Running task')
        Task = self._get_task_class()
        report = TaskReport(self.task_id, Task.get_mode(), dataset)
        task = Task(self.kg_config, self.task_config, dataset, report)
        task.run()
        report.store(self.kg_config['run_id'])
        runtime_in_seconds = (datetime.datetime.now() - start_time).seconds
        get_logger().info(f'Finished task after {runtime_in_seconds} seconds')

    def _get_task_class(self) -> Type[BaseTask]:
        mode = TaskMode(self.task_config['mode'])
        # import modules only on demand to keep dependencies separated
        if mode == TaskMode.CLASSIFICATION:
            from classification_task import ClassificationTask
            return ClassificationTask
        if mode == TaskMode.REGRESSION:
            from regression_task import RegressionTask
            return RegressionTask
        if mode == TaskMode.CLUSTERING:
            from clustering_task import ClusteringTask
            return ClusteringTask
        if mode == TaskMode.DOCUMENT_SIMILARITY:
            from documentsimilarity_task import DocumentSimilarityTask
            return DocumentSimilarityTask
        if mode == TaskMode.ENTITY_RELATEDNESS:
            from entityrelatedness_task import EntityRelatednessTask
            return EntityRelatednessTask
        if mode == TaskMode.SEMANTIC_ANALOGIES:
            from semanticanalogies_task import SemanticAnalogiesTask
            return SemanticAnalogiesTask
        if mode == TaskMode.RECOMMENDATION:
            from recommendation_task import RecommendationTask
            return RecommendationTask
        raise NotImplementedError(f'No task implemented for mode "{mode.value}"')


if __name__ == "__main__":
    task_id = os.environ['KGREAT_STEP']
    with open('config.yaml', mode='r') as f:
        dataset_config = yaml.safe_load(f)
    tm = TaskManager(task_id, dataset_config)
    tm.run_task()
