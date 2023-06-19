from collections import defaultdict
from sklearn.metrics import pairwise_distances
import numpy as np
import pandas as pd
import turicreate as tc
from utils.logging import get_logger
from utils.enums import TaskType, EntityEvalMode
from utils.dataset import RecommendationDataset
from base_task import BaseTask


class RecommendationTask(BaseTask):
    dataset: RecommendationDataset

    @classmethod
    def get_type(cls) -> TaskType:
        return TaskType.RECOMMENDATION

    METRICS = ['F1']
    N_SPLITS = 3
    TOP_K_VALUES = [3, 5, 10]

    def run(self):
        for embedding_type in self.embedding_models:
            get_logger().debug(f'Evaluating recommendation via item similarity for embedding type {embedding_type}')
            related_entities = tc.SFrame(self._get_top_50_related_entities(embedding_type))
            for eval_mode in [EntityEvalMode.KNOWN_ENTITIES, EntityEvalMode.ALL_ENTITIES]:
                use_only_mapped_items = eval_mode == EntityEvalMode.KNOWN_ENTITIES
                actions = tc.SFrame(self.dataset.get_actions(use_only_mapped_items))
                cv_scores = defaultdict(list)
                for i in range(self.N_SPLITS):
                    get_logger().debug(f'Training recommender model for split {i+1}')
                    train_data, val_data = tc.recommender.util.random_split_by_user(actions, user_id='user_id', item_id='item_id', item_test_proportion=0.15)
                    model = self._train_recommender_model(train_data, max(self.TOP_K_VALUES), related_entities)
                    get_logger().debug(f'Evaluating recommender model for split {i+1}')
                    eval_result = model.evaluate_precision_recall(val_data, cutoffs=self.TOP_K_VALUES)['precision_recall_overall']
                    for k in self.TOP_K_VALUES:
                        k_result = eval_result[eval_result['cutoff'] == k]
                        p = k_result['precision'][0]
                        r = k_result['recall'][0]
                        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
                        cv_scores[k].append(f1)
                for k, vals in cv_scores.items():
                    score = float(np.mean(vals))
                    self.report.add_result(eval_mode, 'item_similarity_recommender', {'k': k}, embedding_type, 'F1', score)

    def _train_recommender_model(self, training_data: tc.SFrame, top_k: int, nearest_items: tc.SFrame):
        return tc.item_similarity_recommender.create(training_data, similarity_type='cosine', user_id='user_id', item_id='item_id', target='rating', only_top_k=top_k, nearest_items=nearest_items, target_memory_usage=64*1024**3)

    def _get_top_50_related_entities(self, embedding_type: str) -> pd.DataFrame:
        # compute inter-item similarity via embeddings
        mapped_entities = list(self.dataset.get_mapped_entities())
        mapped_entity_uris = [e[0] for e in mapped_entities]
        mapped_entity_ids = [e[1] for e in mapped_entities]
        entity_features = self.load_entity_embeddings(embedding_type).loc[mapped_entity_uris, :]
        use_cosine = 'use_cosine' in self.task_config and self.task_config['use_cosine']
        entity_similarities = (1-pairwise_distances(entity_features, metric='cosine')) if use_cosine else np.dot(entity_features, entity_features.T)
        top50_related_entities = []
        for entity_id, cosine_scores in zip(mapped_entity_ids, entity_similarities):
            related_ents = np.argsort(-cosine_scores)[:50]
            top50_related_entities.extend([(entity_id, mapped_entity_ids[re], cosine_scores[re]) for re in related_ents])
        return pd.DataFrame(data=top50_related_entities, columns=['item_id', 'similar', 'score'])
