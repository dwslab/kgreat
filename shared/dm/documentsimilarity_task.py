from typing import List, Tuple, Callable, Dict
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from utils.enums import TaskMode, EntityMode
from utils.dataset import DocumentSimilarityDataset
from base_task import BaseTask


# TODO: implement score for EntityMode.ALL_ENTITIES
class DocumentSimilarityTask(BaseTask):
    dataset: DocumentSimilarityDataset

    @classmethod
    def get_mode(cls) -> TaskMode:
        return TaskMode.DOCUMENT_SIMILARITY

    @staticmethod
    def _get_estimators() -> List[Tuple[Callable, dict]]:
        return [
            (DocumentSimilarityTask.document_entity_similarity, {'entity_weights': False}),
            (DocumentSimilarityTask.document_entity_similarity, {'entity_weights': True}),
        ]

    @staticmethod
    def document_entity_similarity(docs_a, docs_b):
        return 1 - pairwise_distances(docs_a, docs_b)

    @staticmethod
    def _get_metrics() -> Dict[str, Callable]:
        return {
            'Spearman': DocumentSimilarityTask._spearman_scorer,
            'Pearson': DocumentSimilarityTask._pearson_scorer,
            'HarmonicMean': DocumentSimilarityTask._harmonic_mean_scorer
        }

    @staticmethod
    def _spearman_scorer(y_true, y_pred) -> float:
        return spearmanr(y_true, y_pred)[0]

    @staticmethod
    def _pearson_scorer(y_true, y_pred) -> float:
        return pearsonr(y_true, y_pred)[0]

    @staticmethod
    def _harmonic_mean_scorer(y_true, y_pred) -> float:
        spearman_score = DocumentSimilarityTask._spearman_scorer(y_true, y_pred)
        pearson_score = DocumentSimilarityTask._pearson_scorer(y_true, y_pred)
        return 2 * spearman_score * pearson_score / (spearman_score + pearson_score)

    def run(self):
        docsim_gold = self.dataset.get_document_similarities()
        for embedding_type in self.embedding_models:
            entity_embeddings = self.load_entity_embeddings(embedding_type)
            for sim_func, params in self._get_estimators():
                docsim_pred = {docs: self._compute_document_similarity(entity_embeddings, docs, sim_func, params) for docs in docsim_gold}
                for metric, metric_scorer in self._get_metrics().items():
                    score = metric_scorer(list(docsim_gold.values()), list(docsim_pred.values()))
                    self.report.add_result(EntityMode.KNOWN_ENTITIES, sim_func.__name__, params, embedding_type, metric, score)

    def _compute_document_similarity(self, entity_embeddings: pd.DataFrame, docs: Tuple[int, int], sim_func: Callable, params: dict) -> float:
        doc1, doc2 = docs
        doc1_ents, doc1_weights = self.dataset.get_mapped_entities_for_document(doc1)
        doc2_ents, doc2_weights = self.dataset.get_mapped_entities_for_document(doc2)
        if not doc1_ents or not doc2_ents:
            return 0
        doc1_entity_embeddings = entity_embeddings.loc[doc1_ents, :]
        doc2_entity_embeddings = entity_embeddings.loc[doc2_ents, :]
        pairwise_similarities = sim_func(doc1_entity_embeddings, doc2_entity_embeddings)
        if params['entity_weights']:
            pairwise_similarities *= np.outer(doc1_weights, doc2_weights)
        max_sim1to2 = np.max(pairwise_similarities, axis=1)
        max_sim2to1 = np.max(pairwise_similarities, axis=0)
        return (sum(max_sim1to2) + sum(max_sim2to1)) / (len(max_sim1to2) + len(max_sim2to1))
