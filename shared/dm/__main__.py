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
from classification_task import ClassificationTask
from regression_task import RegressionTask
from clustering_task import ClusteringTask
from documentsimilarity_task import DocumentSimilarityTask
from entityrelatedness_task import EntityRelatednessTask
from semanticanalogies_task import SemanticAnalogiesTask


class TaskManager:
    AVAILABLE_TASKS = [
        ClassificationTask, RegressionTask, ClusteringTask,
        DocumentSimilarityTask, EntityRelatednessTask, SemanticAnalogiesTask
    ]

    def __init__(self, task_id: str, dataset_config: dict):
        self.task_id = task_id
        self.dataset_config = dataset_config
        self.kg_config = load_kg_config()
        self.task_config = self.kg_config['tasks'][task_id]
        self.tasks_by_mode = {t.get_mode(): t for t in self.AVAILABLE_TASKS}

    def run_task(self):
        start_time = datetime.datetime.now()
        init_logger(self.kg_config['run_id'], self.task_id, self.kg_config['log_level'])
        get_logger().info(f'Starting to run task {self.task_id}')
        # prepare mapping & dataset
        entity_mapping = load_entity_mapping()
        dataset = load_dataset(self.dataset_config, entity_mapping)
        # prepare and run task
        report = TaskReport(self.task_id, self._get_task_class().get_mode(), dataset)
        task = self._get_task_class()(self.kg_config, self.task_config, dataset, report)
        task.run()
        report.store(self.kg_config['run_id'])
        runtime_in_seconds = (datetime.datetime.now() - start_time).seconds
        get_logger().info(f'Finished task after {runtime_in_seconds} seconds')

    def _get_task_class(self) -> Type[BaseTask]:
        mode = TaskMode(self.task_config['mode'])
        return self.tasks_by_mode[mode]


if __name__ == "__main__":
    task_id = os.environ['KGREAT_TASK']
    with open('config.yaml', mode='r') as f:
        dataset_config = yaml.safe_load(f)
    tm = TaskManager(task_id, dataset_config)
    tm.run_task()
