"""Script for creating the dataset based on the two data files from Kaggle"""

import pandas as pd

df_dc = pd.read_csv('dc-wikia-data.csv', header=0)
df_marvel = pd.read_csv('marvel-wikia-data.csv', header=0).rename(columns={'Year': 'YEAR'})
df = pd.concat([df_dc, df_marvel]).rename(columns={'urlslug': 'ComicCharacters_URI'})
df['name'] = df['name'].str.replace('\s*\(.*?\)', '', regex=True)  # remove bracket content (= character origin) in labels

df.to_csv('examples.tsv', sep='\t', header=True, index=False)