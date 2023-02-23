import os
import json
from typing import Type
from utils.io import load_config
from utils.enums import TaskMode
from utils.report import TaskReport
from base_task import BaseTask
from utils.io import load_entity_embeddings, load_entity_mapping
from utils.dataset import load_dataset
from classification_task import ClassificationTask
from regression_task import RegressionTask
from clustering_task import ClusteringTask
from documentsimilarity_task import DocumentSimilarityTask


# TODO: logging
# TODO: improvements through standard scaler?
# TODO: count unmapped entities as errors? (e.g. see Clustering in GEval)


class TaskManager:
    AVAILABLE_TASKS = [ClassificationTask, RegressionTask, ClusteringTask, DocumentSimilarityTask]

    def __init__(self, task_id: str, dataset_config: dict):
        self.task_id = task_id
        self.dataset_config = dataset_config
        self.config = load_config()
        self.task_config = self.config['tasks'][task_id]
        self.tasks_by_mode = {t.get_mode(): t for t in self.AVAILABLE_TASKS}

    def run_task(self):
        # prepare embeddings and dataset
        entity_embeddings = load_entity_embeddings()
        entity_mapping = load_entity_mapping()
        dataset = load_dataset(self.dataset_config, entity_mapping)
        # prepare and run task
        report = TaskReport(self.task_id, self._get_task_class().get_mode().value, dataset)
        task = self._get_task_class()(self.task_config, entity_embeddings, dataset, report)
        task.run()
        report.store(self.config['run_id'])

    def _get_task_class(self) -> Type[BaseTask]:
        mode = TaskMode(self.task_config['mode'])
        return self.tasks_by_mode[mode]


if __name__ == "__main__":
    task_id = os.environ['KGREAT_TASK']
    dataset_config = json.loads(os.environ['KGREAT_DATASET'])
    tm = TaskManager(task_id, dataset_config)
    tm.run_task()
