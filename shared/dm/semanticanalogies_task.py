from typing import List, Optional
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
            for analogy_set in mapped_analogy_sets.itertuples(index=False):
                a, b, c, d = (entity_embeddings.index.get_loc(ent) for ent in analogy_set)  # retrieve ent indices
                d_pred = self._predict_analogy(entity_embeddings, entity_embedding_index, max(self.TOP_K), a, b, c)
                for k in self.TOP_K:
                    if d in d_pred[:k]:
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

    @staticmethod
    def _predict_analogy(entity_embeddings: pd.DataFrame, entity_embedding_index: Optional[hnswlib.Index], max_k: int, a: int, b: int, c: int) -> List[int]:
        d_vec = entity_embeddings.iloc[b, :] - entity_embeddings.iloc[a, :] + entity_embeddings.iloc[c, :]
        if entity_embedding_index is None:
            similarity_to_d = np.dot(entity_embeddings, d_vec)
            most_similar_entities = np.argsort(-similarity_to_d)
        else:
            indices, _ = entity_embedding_index.knn_query(d_vec.to_numpy().reshape(1, -1), k=max_k+3)
            most_similar_entities = indices.flatten()
        return [ent_idx for ent_idx in most_similar_entities if ent_idx not in {a, b, c}]
