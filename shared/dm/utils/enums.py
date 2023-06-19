from enum import Enum


class TaskType(Enum):
    CLASSIFICATION = 'classification'
    REGRESSION = 'regression'
    CLUSTERING = 'clustering'
    DOCUMENT_SIMILARITY = 'documentSimilarity'
    ENTITY_RELATEDNESS = 'entityRelatedness'
    SEMANTIC_ANALOGIES = 'semanticAnalogies'
    RECOMMENDATION = 'recommendation'


class EntityEvalMode(Enum):
    ALL_ENTITIES = 'ALL'  # compute evaluation metrics with respect to *all* entities in the dataset
    KNOWN_ENTITIES = 'KNOWN'  # compute evaluation metrics only with respect to intersection of KG and dataset entities


class DatasetFormat(Enum):
    TSV = 'tsv'
    DOCUMENT_SIMILARITY = 'documentSimilarity'
    ENTITY_RELATEDNESS = 'entityRelatedness'
    SEMANTIC_ANALOGIES = 'semanticAnalogies'
    MOVIELENS_RECOMMENDATION = 'movieLensRecommendation'
    LASTFM_RECOMMENDATION = 'lastFmRecommendation'
    LIBRARYTHING_RECOMMENDATION = 'libraryThingRecommendation'
