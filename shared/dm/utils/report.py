from collections import namedtuple
import csv
from utils.io import get_kg_result_path
from utils.enums import TaskMode, EntityMode
from utils.dataset import Dataset


TaskResult = namedtuple('TaskResult', ['entity_mode', 'estimator', 'estimator_config', 'embedding_type', 'metric', 'score'])


class TaskReport:
    def __init__(self, task_id: str, task_mode: TaskMode, dataset: Dataset):
        self.task_id = task_id
        self.task_mode = task_mode
        self.dataset = dataset
        self.results = []

    def add_result(self, entity_mode: EntityMode, estimator: str, estimator_config: dict, embedding_type: str, metric: str, score: float):
        self.results.append(TaskResult(entity_mode.value, estimator, estimator_config, embedding_type, metric, score))

    def store(self, run_id: str):
        columns = ['id', 'task_mode', 'dataset', 'entities_total', 'entities_missing', 'entity_mode', 'estimator', 'estimator_config', 'embedding_type', 'metric', 'score']
        entities_total = len(self.dataset.get_entities())
        entities_missing = entities_total - len(self.dataset.get_mapped_entities())
        fixed_values = (self.task_id, self.task_mode.value, self.dataset.name, entities_total, entities_missing)

        filepath = get_kg_result_path(run_id) / f'{self.task_id}.tsv'
        with open(filepath, mode='w', newline='') as f:
            writer = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(columns)
            writer.writerows([fixed_values + r for r in self.results])
