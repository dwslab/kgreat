"""Preprocess pre-computed Wikidata embeddings through normalization."""

from pathlib import Path
import pandas as pd
import numpy as np


PATH_TO_EMBEDDINGS_FILE = Path('./embedding/TransE_l1.tsv')


if not PATH_TO_EMBEDDINGS_FILE.is_file():
    raise FileNotFoundError('Could not find the embedding file. Did you forget to run the download script?')
df_embeddings = pd.read_csv(PATH_TO_EMBEDDINGS_FILE, sep='\t', header=None, index_col=0)
df_embeddings = df_embeddings.apply(lambda x: x / np.linalg.norm(x, ord=1), axis=1).round(6)
df_embeddings.to_csv(PATH_TO_EMBEDDINGS_FILE, sep='\t', header=False)
