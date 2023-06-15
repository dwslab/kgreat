from pathlib import Path
import pandas as pd

EMBEDDING_FILE = Path('./embeddings/TransE_l1.tsv')

if not EMBEDDING_FILE.is_file():
    raise FileNotFoundError('Embedding file not found. Run the download script first!')

df = pd.read_csv(EMBEDDING_FILE, sep='\t', header=None, skiprows=1)
df = df[(df[0].str.startswith('<')) & (df[0].str.endswith('>'))]
df[0] = df[0].str.slice(start=1, stop=-1)
df.to_csv(EMBEDDING_FILE, sep='\t', index=False)
