from pathlib import Path
import pandas as pd

EMBEDDING_FILE = Path('./embeddings/TransE_l1.tsv')

if not EMBEDDING_FILE.is_file():
    raise FileNotFoundError('Embedding file not found. Run the download script first!')

data = []
with open(EMBEDDING_FILE) as f:
    for line in f:
        if not line.startswith('<'):
            continue
        item = line.split('\t')
        if not item[0].endswith('>'):
            continue
        item[0] = item[0][1:-1]
        data.append(item)
pd.DataFrame(data=data, columns=['ent'] + list(range(200))).to_csv(EMBEDDING_FILE, sep='\t', index=False, header=False)
