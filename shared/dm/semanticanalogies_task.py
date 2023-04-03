from typing import List
from collections import defaultdict
import numpy as np
from utils.enums import TaskMode, EntityMode
from utils.dataset import SemanticAnalogiesDataset
from base_task import BaseTask


class SemanticAnalogiesTask(BaseTask):
    dataset: SemanticAnalogiesDataset
    TOP_K = [2, 5]

    @classmethod
    def get_mode(cls) -> TaskMode:
        return TaskMode.SEMANTIC_ANALOGIES

    def run(self):
        mapped_analogy_sets = self.dataset.get_entity_analogy_sets(True)
        correct_predictions_by_k = defaultdict(int)
        for analogy_set in mapped_analogy_sets.itertuples(index=False):
            a, b, c, d = self.entity_embeddings.index[analogy_set]  # retrieve entity indices for analogy set
            d_pred = self._predict_analogy(a, b, c)
            for k in self.TOP_K:
                if d in d_pred[:k]:
                    correct_predictions_by_k[k] += 1
        # report results for all/known entities
        eval_scenarios = [
            (EntityMode.ALL_ENTITIES, len(self.dataset.get_entity_analogy_sets(False))),
            (EntityMode.KNOWN_ENTITIES, len(mapped_analogy_sets))
        ]
        for entity_mode, total_entity_count in eval_scenarios:
            for k in self.TOP_K:
                score = correct_predictions_by_k[k] / total_entity_count
                self.report.add_result(entity_mode, 'Cosine similarity', {'top_k': k}, 'Accuracy', score)

    def _predict_analogy(self, a: int, b: int, c: int) -> List[int]:
        d_vec = self.entity_embeddings.iloc[b, :] - self.entity_embeddings.iloc[a, :] + self.entity_embeddings.iloc[c, :]
        similarity_to_d = np.dot(self.entity_embeddings, d_vec)
        return [ent_idx for ent_idx in np.argsort(-similarity_to_d) if ent_idx not in {a, b, c}]
