from typing import List, Dict, Tuple
from abc import ABC, abstractmethod
import json
import pandas as pd
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
    def get_entities(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_mapped_entities(self) -> set:
        pass

    @abstractmethod
    def get_entity_labels(self) -> pd.Series:
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
        # apply mapping to entities
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
        return set(self.mapped_data)

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
        # apply mapping to entities
        for doc_id, ents in self.document_entities.items():
            mapped_doc_ents = {self.entity_mapping[e]: w for e, w in ents.items() if e in self.entity_mapping}
            self.mapped_document_entities[doc_id] = mapped_doc_ents

    def get_entities(self) -> pd.DataFrame:
        return pd.DataFrame({k: [e for ents in self.document_entities.values() for e in ents] for k in self.entity_keys}).drop_duplicates()

    def get_mapped_entities(self) -> set:
        return {e for ents in self.mapped_document_entities.values() for e in ents}

    def get_entity_labels(self) -> pd.Series:
        raise NotImplementedError('Method not implemented for DocumentSimilarity task.')

    def get_document_ids(self) -> List[int]:
        return list(self.document_entities)

    def get_mapped_entities_for_document(self, document_id: int) -> Tuple[List[str], List[float]]:
        entities_with_weights = self.mapped_document_entities[document_id]
        return list(entities_with_weights), list(entities_with_weights.values())

    def get_document_similarities(self) -> Dict[Tuple[int, int], float]:
        return self.document_similarities


def load_dataset(config: dict, entity_mapping: pd.DataFrame) -> Dataset:
    dataset_by_format = {ds.get_format(): ds for ds in Dataset.__subclasses__()}
    dataset_format = DatasetFormat(config['format'])
    dataset = dataset_by_format[dataset_format](config, entity_mapping)
    dataset.load()
    return dataset
