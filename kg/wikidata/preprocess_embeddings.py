import pandas as pd
import numpy as np


PATH_TO_EMBEDDINGS_FILE = './embeddings/TransE_l1.tsv'


df = pd.read_csv(PATH_TO_EMBEDDINGS_FILE, sep='\t', header=None, index_col=0)
df = df.apply(lambda x: x / np.linalg.norm(x, ord=1), axis=1)  # normalize to unit vectors
df.to_csv(PATH_TO_EMBEDDINGS_FILE, sep='\t', header=False)
