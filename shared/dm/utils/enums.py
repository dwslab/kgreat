from enum import Enum


class TaskMode(Enum):
    CLASSIFICATION = 'classification'
    REGRESSION = 'regression'
    CLUSTERING = 'clustering'
    DOCUMENT_SIMILARITY = 'document_similarity'
    ENTITY_RELATEDNESS = 'entity_relatedness'
    SEMANTIC_ANALOGIES = 'semantic_analogies'


class DatasetFormat(Enum):
    TSV = 'tsv'
