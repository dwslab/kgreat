from typing import List
from collections import defaultdict
import numpy as np
import pandas as pd
import hnswlib
from utils.logging import get_logger
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
        for embedding_type in self.embedding_models:
            correct_predictions_by_k = defaultdict(int)
            entity_embeddings = self.load_entity_embeddings(embedding_type)
            entity_embedding_index = self._build_ann_index(entity_embeddings.to_numpy(), 400, 64, 50)
            get_logger().debug(f'Evaluating semantic analogies via cosine distance for embedding type {embedding_type}')
            for analogy_set in mapped_analogy_sets.itertuples(index=False):
                a, b, c, d = (entity_embeddings.index.get_loc(ent) for ent in analogy_set)  # retrieve ent indices
                d_pred = self._predict_analogy(entity_embeddings, entity_embedding_index, max(self.TOP_K), a, b, c)
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
                    score = correct_predictions_by_k[k] / total_entity_count if total_entity_count > 0 else 0
                    self.report.add_result(entity_mode, 'Cosine similarity', {'top_k': k}, embedding_type, 'Accuracy', score)

    @staticmethod
    def _build_ann_index(embeddings: np.ndarray, ef_construction: int, M: int, ef: int) -> hnswlib.Index:
        get_logger().debug('Building ANN index..')
        index = hnswlib.Index(space='ip', dim=embeddings.shape[-1])
        index.init_index(max_elements=len(embeddings), ef_construction=ef_construction, M=M)
        index.add_items(embeddings, list(range(len(embeddings))))
        index.set_ef(ef)
        return index

    @staticmethod
    def _predict_analogy(entity_embeddings: pd.DataFrame, entity_embedding_index: hnswlib.Index, max_k: int, a: int, b: int, c: int) -> List[int]:
        d_vec = entity_embeddings.iloc[b, :] - entity_embeddings.iloc[a, :] + entity_embeddings.iloc[c, :]
        indices, _ = entity_embedding_index.knn_query(d_vec.to_numpy().reshape(1, -1), k=max_k+3)
        return [ent_idx for ent_idx in indices.flatten() if ent_idx not in {a, b, c}]
