from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.stats import kendalltau
import random
from utils.logging import get_logger
from utils.enums import TaskMode, EntityMode
from utils.dataset import EntityRelatednessDataset
from base_task import BaseTask


class EntityRelatednessTask(BaseTask):
    dataset: EntityRelatednessDataset

    @classmethod
    def get_mode(cls) -> TaskMode:
        return TaskMode.ENTITY_RELATEDNESS

    def run(self):
        # compute similarity of main entities to their related entities (if known)
        known_ents = self.dataset.get_mapped_entity_relations()
        for embedding_type in self.embedding_models:
            entity_embeddings = self.load_entity_embeddings(embedding_type)
            get_logger().debug(f'Evaluating entity relatedness via cosine distance for embedding type {embedding_type}')
            known_ents_rankings = [self._compute_entity_similarities(entity_embeddings, main_ent, rel_ents) for main_ent, rel_ents in known_ents]
            # append unknown entities in random order for similarities over all entities
            all_ents_indices = [set(range(len(rel_ents))) for _, rel_ents in self.dataset.get_entity_relations()]
            unknown_ent_indices = [all_ents.difference(set(known_ents)) for all_ents, known_ents in zip(all_ents_indices, known_ents_rankings)]
            all_ents_rankings = [np.append(known_ents, random.sample(unknown_ents, len(unknown_ents))) for known_ents, unknown_ents in zip(known_ents_rankings, unknown_ent_indices)]
            # evaluate similarities for all/known entities
            eval_scenarios = [
                (EntityMode.ALL_ENTITIES, all_ents_rankings),
                (EntityMode.KNOWN_ENTITIES, known_ents_rankings)
            ]
            for entity_mode, entity_rankings in eval_scenarios:
                score = self._evaluate_entity_rankings(entity_rankings)
                self.report.add_result(entity_mode, 'Cosine distance', {}, embedding_type, 'Kendall\'s tau', score)

    def _compute_entity_similarities(self, entity_embeddings: pd.DataFrame, main_ent: Optional[str], related_entities: Dict[str, int]) -> List[int]:
        if not main_ent or not related_entities:
            return []
        rel_ent_names, rel_ent_indices = list(related_entities), np.array(list(related_entities.values()))
        # find entity ranking through embedding distance
        main_entity_embedding = entity_embeddings.loc[[main_ent]]
        related_entities_embeddings = entity_embeddings.loc[rel_ent_names]
        entity_distances = distance.cdist(main_entity_embedding, related_entities_embeddings, 'cosine').flatten()
        return rel_ent_indices[entity_distances.argsort()]

    def _evaluate_entity_rankings(self, entity_rankings: List[List[int]]) -> float:
        scores_per_main_entity = []
        for entity_ranking in entity_rankings:
            if not entity_ranking:
                continue  # skip empty rankings (e.g., in the case of completely unmapped sets of entities)
            scores_per_main_entity.append(kendalltau(sorted(entity_ranking), entity_ranking)[0])
        return np.average(scores_per_main_entity)
