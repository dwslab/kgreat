import pandas as pd
from abc import ABC, abstractmethod
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


def load_dataset(config: dict, entity_mapping: pd.DataFrame) -> Dataset:
    dataset_by_format = {ds.get_format(): ds for ds in Dataset.__subclasses__()}
    dataset_format = DatasetFormat(config['format'])
    dataset = dataset_by_format[dataset_format](config, entity_mapping)
    dataset.load()
    return dataset
