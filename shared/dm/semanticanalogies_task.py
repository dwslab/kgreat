from typing import Optional, Iterable
from collections import defaultdict
import numpy as np
import pandas as pd
import hnswlib
from utils.logger import get_logger
from utils.enums import TaskType, EntityEvalMode
from utils.dataset import SemanticAnalogiesDataset
from utils.io import get_embedding_path
from base_task import BaseTask


class SemanticAnalogiesTask(BaseTask):
    dataset: SemanticAnalogiesDataset
    TOP_K = [2, 5]

    @classmethod
    def get_type(cls) -> TaskType:
        return TaskType.SEMANTIC_ANALOGIES

    def run(self):
        mapped_analogy_sets = self.dataset.get_entity_analogy_sets(True)
        for embedding_type in self.embedding_models:
            correct_predictions_by_k = defaultdict(int)
            entity_embeddings = self.load_entity_embeddings(embedding_type, False)
            entity_embedding_index = self._load_entity_embedding_index(embedding_type)
            get_logger().debug(f'Evaluating semantic analogies via cosine distance for embedding type {embedding_type}')
            analogy_sets_vecs = np.array([[entity_embeddings.index.get_loc(ent) for ent in analogy_set] for analogy_set in mapped_analogy_sets.itertuples(index=False)])
            analogy_sets_pred = analogy_sets_vecs[:, 1] - analogy_sets_vecs[:, 0] + analogy_sets_vecs[:, 2]
            for d_preds, d_true in self._predict_analogies(entity_embeddings, entity_embedding_index, max(self.TOP_K), analogy_sets_vecs, analogy_sets_pred):
                for k in self.TOP_K:
                    if d_true in d_preds[:k]:
                        correct_predictions_by_k[k] += 1
            # report results for all/known entities
            eval_scenarios = [
                (EntityEvalMode.ALL_ENTITIES, len(self.dataset.get_entity_analogy_sets(False))),
                (EntityEvalMode.KNOWN_ENTITIES, len(mapped_analogy_sets))
            ]
            for eval_mode, total_entity_count in eval_scenarios:
                for k in self.TOP_K:
                    score = correct_predictions_by_k[k] / total_entity_count if total_entity_count > 0 else 0
                    self.report.add_result(eval_mode, 'Cosine similarity', {'top_k': k}, embedding_type, 'Accuracy', score)

    @staticmethod
    def _load_entity_embedding_index(embedding_type: str) -> Optional[hnswlib.Index]:
        filepath = get_embedding_path(ann=True) / f'{embedding_type}_index.p'
        if not filepath.is_file():
            return None
        return hnswlib.Index(space='ip', dim=200).load_index(str(filepath))

    def _predict_analogies(self, entity_embeddings: pd.DataFrame, index: Optional[hnswlib.Index], top_k: int, analogy_sets_vecs: np.array, analogy_sets_preds: np.array) -> Iterable:
        if index is None:
            for (a, b, c, d), d_similarities in zip([analogy_sets_vecs, np.dot(entity_embeddings, analogy_sets_preds)]):
                d_preds = np.argsort(-d_similarities)
                yield [ent_idx for ent_idx in d_preds if ent_idx not in {a, b, c}], d
        else:
            for analogy_idx, (entity_indices, _) in enumerate(zip(*index.knn_query(analogy_sets_preds, k=top_k, num_threads=self.kg_config['max_cpus']))):
                a, b, c, d = analogy_sets_vecs[analogy_idx]
                yield [ent_idx for ent_idx in entity_indices if ent_idx not in {a, b, c}], d
