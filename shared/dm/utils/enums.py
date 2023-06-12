from enum import Enum


class TaskMode(Enum):
    CLASSIFICATION = 'classification'
    REGRESSION = 'regression'
    CLUSTERING = 'clustering'
    DOCUMENT_SIMILARITY = 'documentSimilarity'
    ENTITY_RELATEDNESS = 'entityRelatedness'
    SEMANTIC_ANALOGIES = 'semanticAnalogies'
    RECOMMENDATION = 'recommendation'


class EntityMode(Enum):
    ALL_ENTITIES = 'ALL'
    KNOWN_ENTITIES = 'KNOWN'


class DatasetFormat(Enum):
    TSV = 'tsv'
    DOCUMENT_SIMILARITY = 'documentSimilarity'
    ENTITY_RELATEDNESS = 'entityRelatedness'
    SEMANTIC_ANALOGIES = 'semanticAnalogies'
    MOVIELENS_RECOMMENDATION = 'movieLensRecommendation'
    LASTFM_RECOMMENDATION = 'lastFmRecommendation'
    LIBRARYTHING_RECOMMENDATION = 'libraryThingRecommendation'
