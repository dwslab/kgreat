from typing import List, Dict, Tuple, Optional
from abc import ABC, abstractmethod
from collections import defaultdict
import json
import pandas as pd
from .enums import DatasetFormat
from utils.io import load_entity_embeddings


class Dataset(ABC):
    def __init__(self, dataset_config: dict, kg_config: dict, entity_mapping: pd.DataFrame):
        self.name = dataset_config['name']
        self.entity_keys = dataset_config['entity_keys']
        # create dict-like mapping from any possible URI in this dataset to the source
        valid_entities = set(load_entity_embeddings(kg_config['preprocessing']['embeddings']['models'][0]).index.values)
        self.entity_mapping = {}
        for key in self.entity_keys:
            if key not in entity_mapping:
                continue
            for k, v in entity_mapping[[key, 'source']].itertuples(index=False, name=None):
                if v in valid_entities:
                    self.entity_mapping[k] = v

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

    def get_label_from_dbpedia_uri(self, dbpedia_uri: str) -> str:
        return dbpedia_uri[len('http://dbpedia.org/resource/'):].replace('_', ' ')


class TsvDataset(Dataset):
    def __init__(self, dataset_config: dict, kg_config: dict, entity_mapping: pd.DataFrame):
        super().__init__(dataset_config, kg_config, entity_mapping)
        self.entity_label = dataset_config['entity_label'] if 'entity_label' in dataset_config else None
        self.data_file = dataset_config['data_file']
        self.label_column = dataset_config['label']
        self.data = None
        self.mapped_data = None
        self.unmapped_labels = None

    @classmethod
    def get_format(cls) -> DatasetFormat:
        return DatasetFormat.TSV

    def load(self):
        valid_columns = self.entity_keys + [self.label_column]
        if self.entity_label:
            valid_columns.append(self.entity_label)
        self.data = pd.read_csv(self.data_file, sep='\t', header=0, index_col=None, usecols=valid_columns)

    def apply_mapping(self):
        mapped_data = {}
        unmapped_labels = []
        for _, row in self.data.iterrows():
            row_mapped = False
            for key in self.entity_keys:
                if key not in row or row[key] not in self.entity_mapping:
                    continue
                source_key = self.entity_mapping[row[key]]
                mapped_data[source_key] = row[self.label_column]
                row_mapped = True
            if not row_mapped:
                unmapped_labels.append(row[self.label_column])
        self.mapped_data = pd.Series(mapped_data)
        self.unmapped_labels = unmapped_labels

    def get_entities(self) -> pd.DataFrame:
        if self.entity_label:
            all_keys = self.entity_keys + [self.entity_label]
            df = self.data[all_keys].drop_duplicates(subset=self.entity_keys).rename(columns={self.entity_label: 'label'})
        else:
            df = self.data[self.entity_keys].drop_duplicates()
            df['label'] = df['DBpedia16_URI'].apply(self.get_label_from_dbpedia_uri)
        return df

    def get_mapped_entities(self) -> set:
        return set(self.mapped_data.index)

    def get_entity_labels(self, mapped: bool = True) -> pd.Series:
        return self.mapped_data if mapped else self.unmapped_labels


class DocumentSimilarityDataset(Dataset):
    def __init__(self, dataset_config: dict, kg_config: dict, entity_mapping: pd.DataFrame):
        super().__init__(dataset_config, kg_config, entity_mapping)
        self.entity_file = dataset_config['entity_file']
        self.docsim_file = dataset_config['docsim_file']
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
        df = pd.DataFrame({k: [e for ents in self.document_entities.values() for e in ents] for k in self.entity_keys}).drop_duplicates()
        df['label'] = df['DBpedia16_URI'].apply(self.get_label_from_dbpedia_uri)
        return df

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
    def __init__(self, dataset_config: dict, kg_config: dict, entity_mapping: pd.DataFrame):
        super().__init__(dataset_config, kg_config, entity_mapping)
        self.data_file = dataset_config['data_file']
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
        df = pd.DataFrame({k: list(ents) for k in self.entity_keys}).drop_duplicates()
        df['label'] = df['DBpedia16_URI'].apply(self.get_label_from_dbpedia_uri)
        return df

    def get_mapped_entities(self) -> set:
        return {me for me, _ in self.mapped_data if me is not None} | {re for _, rel_ents in self.data for re in rel_ents}

    def get_entity_relations(self) -> List[Tuple[str, List[str]]]:
        return self.data

    def get_mapped_entity_relations(self) -> List[Tuple[Optional[str], Dict[str, int]]]:
        return self.mapped_data


class SemanticAnalogiesDataset(Dataset):
    def __init__(self, dataset_config: dict, kg_config: dict, entity_mapping: pd.DataFrame):
        super().__init__(dataset_config, kg_config, entity_mapping)
        self.data_file = dataset_config['data_file']
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
        df = pd.DataFrame({k: ents for k in self.entity_keys}).drop_duplicates()
        df['label'] = df['DBpedia16_URI'].apply(self.get_label_from_dbpedia_uri)
        return df

    def get_mapped_entities(self) -> set:
        return set().union(*[self.mapped_data[col] for col in self.mapped_data])

    def get_entity_analogy_sets(self, mapped: bool) -> pd.DataFrame:
        return self.mapped_data if mapped else self.data


class RecommendationDataset(Dataset, ABC):
    def __init__(self, dataset_config: dict, kg_config: dict, entity_mapping: pd.DataFrame):
        super().__init__(dataset_config, kg_config, entity_mapping)
        self.item_file = dataset_config['item_file']
        self.action_file = dataset_config['action_file']
        self.dbplink_file = dataset_config['dbplink_file']
        self.items = None
        self.mapped_items = None
        self.actions = None

    def postprocess_data(self):
        self.actions['item_count'] = self.actions['item_id'].map(self.actions['item_id'].value_counts())
        self.actions['user_count'] = self.actions['user_id'].map(self.actions['user_id'].value_counts())
        # remove most popular items (top 1%)
        one_percent_of_items = int(self.actions['item_id'].nunique() * 0.01)
        top_items = self.actions['item_id'].value_counts().nlargest(n=one_percent_of_items).index.values
        self.actions = self.actions[~self.actions['item_id'].isin(top_items)]
        # remove items and users with too few ratings (less than five)
        self.actions = self.actions[(self.actions['item_count'] >= 5) & (self.actions['user_count'] >= 5)]
        self.actions = self.actions.drop(columns=['item_count', 'user_count'])
        # remove unrated items
        self.items = self.items[self.items.index.isin(self.actions['item_id'].astype(int).unique())]

    def apply_mapping(self):
        mapped_items = {}
        for item_id, row in self.items.iterrows():
            for key in self.entity_keys:
                if key not in row or row[key] not in self.entity_mapping:
                    continue
                source_key = self.entity_mapping[row[key]]
                mapped_items[source_key] = item_id
        self.mapped_items = pd.Series(mapped_items)

    def get_entities(self) -> pd.DataFrame:
        return self.items

    def get_mapped_entities(self) -> set:
        return set(zip(self.mapped_items.index, self.mapped_items))

    def get_actions(self, mapped: bool = True) -> pd.DataFrame:
        return self.actions[self.actions['item_id'].isin(self.mapped_items)] if mapped else self.actions


class MovieLensRecommendationDataset(RecommendationDataset):
    def __init__(self, dataset_config: dict, kg_config: dict, entity_mapping: pd.DataFrame):
        super().__init__(dataset_config, kg_config, entity_mapping)
        self.link_file = dataset_config['link_file']

    @classmethod
    def get_format(cls) -> DatasetFormat:
        return DatasetFormat.MOVIELENS_RECOMMENDATION

    def load(self):
        # initialize movies
        movie_labels = pd.read_csv(self.item_file, sep=',', header=0, index_col=0)['title'].rename('label')
        imdb_links = pd.read_csv(self.link_file, sep=',', header=0, index_col=0)
        imdb_links['imdb_URI'] = imdb_links['imdbId'].apply(lambda imdb_id: f'https://www.imdb.com/title/tt{imdb_id:07}/')
        movies = pd.merge(movie_labels, imdb_links[['imdb_URI']], how='left', left_index=True, right_index=True)
        dbp_links = pd.read_csv(self.dbplink_file, sep='\t', index_col=0, names=['title', 'DBpedia16_URI'])
        movies = pd.merge(movies, dbp_links['DBpedia16_URI'], how='left', left_index=True, right_index=True)
        self.items = movies
        # initialize movie ratings
        ratings = pd.read_csv(self.action_file, sep=',', header=0, usecols=['userId', 'movieId', 'rating'])
        self.actions = ratings.rename(columns={'userId': 'user_id', 'movieId': 'item_id'})
        self.postprocess_data()


class LastFmRecommendationDataset(RecommendationDataset):
    @classmethod
    def get_format(cls) -> DatasetFormat:
        return DatasetFormat.LASTFM_RECOMMENDATION

    def load(self):
        # initialize artists
        artists = pd.read_csv(self.item_file, sep='\t', header=0, index_col=0)
        artists = artists.drop(columns='pictureURL').rename(columns={'name': 'label', 'url': 'lastfm_URI'})
        dbp_links = pd.read_csv(self.dbplink_file, sep='\t', index_col=0, names=['title', 'DBpedia16_URI'])
        artists = pd.merge(artists, dbp_links['DBpedia16_URI'], how='left', left_index=True, right_index=True)
        self.items = artists
        # initialize artist listening counts
        listening_counts = pd.read_csv(self.action_file, sep='\t', header=0)
        self.actions = listening_counts.rename(columns={'userID': 'user_id', 'artistID': 'item_id', 'weight': 'rating'})
        self.postprocess_data()


class LibraryThingRecommendationDataset(RecommendationDataset):
    @classmethod
    def get_format(cls) -> DatasetFormat:
        return DatasetFormat.LIBRARYTHING_RECOMMENDATION

    def load(self):
        # initialize books
        books = pd.read_csv(self.item_file, sep='\t', header=0)
        books['LibraryThing_URI'] = books['item_id'].apply(lambda item_id: f'https://www.librarything.com/work/{item_id}')
        books = books.set_index('item_id', drop=True)
        dbp_links = pd.read_csv(self.dbplink_file, sep='\t', index_col=0, names=['title', 'DBpedia16_URI'])
        self.items = pd.merge(books, dbp_links['DBpedia16_URI'], how='left', left_index=True, right_index=True)
        # initialize user reviews
        actions = {}
        with open(self.action_file) as f:
            for line in f:
                data = eval(line[line.find(' = ') + 3:])
                if 'user' not in data or 'work' not in data or 'stars' not in data:
                    continue
                actions[(data['user'], int(data['work']))] = float(data['stars'])
        actions = [(user, work, rating) for (user, work), rating in actions.items()]
        self.actions = pd.DataFrame(data=actions, columns=['user_id', 'item_id', 'rating'])
        self.postprocess_data()


def load_dataset(dataset_config: dict, kg_config: dict, entity_mapping: pd.DataFrame) -> Dataset:
    dataset_by_format = {ds.get_format(): ds for ds in _get_transitive_subclasses(Dataset)}
    dataset_format = DatasetFormat(dataset_config['format'])
    dataset = dataset_by_format[dataset_format](dataset_config, kg_config, entity_mapping)
    dataset.load()
    if len(entity_mapping):
        dataset.apply_mapping()
    return dataset


def _get_transitive_subclasses(cls):
    for subcls in cls.__subclasses__():
        if subcls.__subclasses__():
            yield from _get_transitive_subclasses(subcls)
        else:
            yield subcls
