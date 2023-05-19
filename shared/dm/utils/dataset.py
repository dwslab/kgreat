from typing import List, Dict, Tuple, Optional
from abc import ABC, abstractmethod
from collections import defaultdict
import json
import pandas as pd
from utils.logging import get_logger
from utils.enums import DatasetFormat


class Dataset(ABC):
    def __init__(self, config: dict, entity_mapping: pd.DataFrame):
        self.name = config['name']
        self.entity_keys = config['entity_keys']
        # create dict-like mapping from any possible URI in this dataset to the source
        self.entity_mapping = {}
        for key in self.entity_keys:
            if key not in entity_mapping:
                continue
            self.entity_mapping |= entity_mapping.set_index(key)['source'].to_dict()

    @classmethod
    @abstractmethod
    def get_format(cls) -> DatasetFormat:
        pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def apply_mapping(self):
        pass

    @abstractmethod
    def get_entities(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_mapped_entities(self) -> set:
        pass


class TsvDataset(Dataset):
    def __init__(self, config: dict, entity_mapping: pd.DataFrame):
        super().__init__(config, entity_mapping)
        self.data_file = config['data_file']
        self.label_column = config['label']
        self.data = None
        self.mapped_data = None

    @classmethod
    def get_format(cls) -> DatasetFormat:
        return DatasetFormat.TSV

    def load(self):
        valid_columns = self.entity_keys + [self.label_column]
        self.data = pd.read_csv(self.data_file, sep='\t', header=0, index_col=None, usecols=valid_columns)

    def apply_mapping(self):
        mapped_data = {}
        for _, row in self.data.iterrows():
            for key in self.entity_keys:
                if key not in row or row[key] not in self.entity_mapping:
                    continue
                source_key = self.entity_mapping[row[key]]
                mapped_data[source_key] = row[self.label_column]
        self.mapped_data = pd.Series(mapped_data)

    def get_entities(self) -> pd.DataFrame:
        return self.data[self.entity_keys].drop_duplicates()

    def get_mapped_entities(self) -> set:
        return set(self.mapped_data.index)

    def get_entity_labels(self) -> pd.Series:
        return self.mapped_data


class DocumentSimilarityDataset(Dataset):
    def __init__(self, config: dict, entity_mapping: pd.DataFrame):
        super().__init__(config, entity_mapping)
        self.entity_file = config['entity_file']
        self.docsim_file = config['docsim_file']
        self.document_entities = {}
        self.mapped_document_entities = {}
        self.document_similarities = {}

    @classmethod
    def get_format(cls) -> DatasetFormat:
        return DatasetFormat.DOCUMENT_SIMILARITY

    def load(self):
        # load document entities
        with open(self.entity_file) as f:
            doc_entities_data = json.load(f)
        for i, doc_data in enumerate(doc_entities_data, start=1):
            self.document_entities[i] = {ent_data['entity']: ent_data['weight'] for ent_data in doc_data['annotations']}
        # load document similarities
        for doc1, doc2, sim in pd.read_csv(self.docsim_file, sep=',', header=0).itertuples(index=False):
            docs_key = tuple(sorted((doc1, doc2)))
            self.document_similarities[docs_key] = sim

    def apply_mapping(self):
        for doc_id, ents in self.document_entities.items():
            mapped_doc_ents = {self.entity_mapping[e]: w for e, w in ents.items() if e in self.entity_mapping}
            self.mapped_document_entities[doc_id] = mapped_doc_ents

    def get_entities(self) -> pd.DataFrame:
        return pd.DataFrame({k: [e for ents in self.document_entities.values() for e in ents] for k in self.entity_keys}).drop_duplicates()

    def get_mapped_entities(self) -> set:
        return {e for ents in self.mapped_document_entities.values() for e in ents}

    def get_document_ids(self) -> List[int]:
        return list(self.document_entities)

    def get_mapped_entities_for_document(self, document_id: int) -> Tuple[List[str], List[float]]:
        entities_with_weights = self.mapped_document_entities[document_id]
        return list(entities_with_weights), list(entities_with_weights.values())

    def get_document_similarities(self) -> Dict[Tuple[int, int], float]:
        return self.document_similarities


class EntityRelatednessDataset(Dataset):
    def __init__(self, config: dict, entity_mapping: pd.DataFrame):
        super().__init__(config, entity_mapping)
        self.data_file = config['data_file']
        self.data = []
        self.mapped_data = []

    @classmethod
    def get_format(cls) -> DatasetFormat:
        return DatasetFormat.ENTITY_RELATEDNESS

    def load(self):
        # load entities and their related entities
        entity_relations = defaultdict(list)
        current_main_ent = None
        for main_ent, related_ent in pd.read_csv(self.data_file, sep='\t', header=0).itertuples(index=False):
            if isinstance(main_ent, str):
                current_main_ent = main_ent
            elif isinstance(related_ent, str):
                entity_relations[current_main_ent].append(related_ent)
        self.data = [(main_ent, rel_ents) for main_ent, rel_ents in entity_relations.items()]

    def apply_mapping(self):
        for ent, rel_ents in self.data:
            mapped_main_ent = self.entity_mapping[ent] if ent in self.entity_mapping else None
            mapped_rel_ents = {self.entity_mapping[e]: idx for idx, e in enumerate(rel_ents) if e in self.entity_mapping}
            self.mapped_data.append((mapped_main_ent, mapped_rel_ents))

    def get_entities(self) -> pd.DataFrame:
        ents = {me for me, _ in self.data} | {re for _, rel_ents in self.data for re in rel_ents}
        return pd.DataFrame({k: list(ents) for k in self.entity_keys})

    def get_mapped_entities(self) -> set:
        return {me for me, _ in self.mapped_data if me is not None} | {re for _, rel_ents in self.data for re in rel_ents}

    def get_entity_relations(self) -> List[Tuple[str, List[str]]]:
        return self.data

    def get_mapped_entity_relations(self) -> List[Tuple[Optional[str], Dict[str, int]]]:
        return self.mapped_data


class SemanticAnalogiesDataset(Dataset):
    def __init__(self, config: dict, entity_mapping: pd.DataFrame):
        super().__init__(config, entity_mapping)
        self.data_file = config['data_file']
        self.data = None
        self.mapped_data = None

    @classmethod
    def get_format(cls) -> DatasetFormat:
        return DatasetFormat.SEMANTIC_ANALOGIES

    def load(self):
        self.data = pd.read_csv(self.data_file, sep='\t', header=None, index_col=None, names=['a', 'b', 'c', 'd'])

    def apply_mapping(self):
        # remove any quadruple if at least one entity can't be mapped
        self.mapped_data = self.data.applymap(lambda x: self.entity_mapping.get(x)).dropna(how='any', axis=0)

    def get_entities(self) -> pd.DataFrame:
        ents = list(set().union(*[self.data[col] for col in self.data]))
        return pd.DataFrame({k: ents for k in self.entity_keys}).drop_duplicates()

    def get_mapped_entities(self) -> set:
        return set().union(*[self.mapped_data[col] for col in self.mapped_data])

    def get_entity_analogy_sets(self, mapped: bool) -> pd.DataFrame:
        return self.mapped_data if mapped else self.data


def load_dataset(config: dict, entity_mapping: pd.DataFrame) -> Dataset:
    get_logger().info('Loading dataset')
    dataset_by_format = {ds.get_format(): ds for ds in Dataset.__subclasses__()}
    dataset_format = DatasetFormat(config['format'])
    dataset = dataset_by_format[dataset_format](config, entity_mapping)
    dataset.load()
    if len(entity_mapping):
        get_logger().info('Applying entity mapping')
        dataset.apply_mapping()
    return dataset
