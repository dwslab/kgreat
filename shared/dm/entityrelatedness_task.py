from typing import List, Dict
import numpy as np
from scipy.spatial import distance
from scipy.stats import kendalltau
import random
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
        known_ents = self.dataset.get_entities_with_related_entities(True)
        known_ents_rankings = [self._compute_entity_similarities(main_ent, rel_ents) for main_ent, rel_ents in known_ents.items()]
        # append unknown entities in random order for similarities over all entities
        all_ents_indices = [set(ents.values()) for ents in self.dataset.get_entities_with_related_entities(False).values()]
        unknown_ent_indices = [all_ents.difference(set(known_ents)) for all_ents, known_ents in zip(all_ents_indices, known_ents_rankings)]
        all_ents_rankings = [known_ents + random.sample(unknown_ents, len(unknown_ents)) for known_ents, unknown_ents in zip(known_ents_rankings, unknown_ent_indices)]
        # evaluate similarities for all/known entities
        eval_scenarios = [
            (EntityMode.ALL_ENTITIES, all_ents_rankings),
            (EntityMode.KNOWN_ENTITIES, known_ents_rankings)
        ]
        for entity_mode, entity_rankings in eval_scenarios:
            score = self._evaluate_entity_rankings(entity_rankings)
            self.report.add_result(entity_mode, 'Cosine distance', {}, 'Kendall\'s tau', score)

    def _compute_entity_similarities(self, main_ent: str, related_entities: Dict[str, int]) -> List[int]:
        if not related_entities:
            return []
        rel_ent_names, rel_ent_indices = list(related_entities), np.array(list(related_entities.values()))
        # find entity ranking through embedding distance
        main_entity_embedding = self.entity_embeddings.loc[[main_ent]]
        related_entities_embeddings = self.entity_embeddings.loc[rel_ent_names]
        entity_distances = distance.cdist(main_entity_embedding, related_entities_embeddings, 'cosine').flatten()
        return rel_ent_indices[entity_distances.argsort()]

    def _evaluate_entity_rankings(self, entity_rankings: List[List[int]]) -> float:
        scores_per_main_entity = []
        for entity_ranking in entity_rankings:
            if not entity_ranking:
                continue  # skip empty rankings (e.g., in the case of completely unmapped sets of entities)
            scores_per_main_entity.append(kendalltau(sorted(entity_ranking), entity_ranking)[0])
        return np.average(scores_per_main_entity)
