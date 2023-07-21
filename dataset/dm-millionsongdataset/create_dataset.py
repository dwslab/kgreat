import pandas as pd
URL_TRACKS = 'http://millionsongdataset.com/sites/default/files/AdditionalFiles/unique_tracks.txt'
URL_GENRES = 'http://www.ifs.tuwien.ac.at/mir/msd/partitions/msd-topMAGD-genreAssignment.cls'

df_tracks = pd.read_csv(URL_TRACKS, sep='<SEP>', header=None, index_col=None, names=['msd_URI', 'artist', 'title'], usecols=[0,2,3])
df_genres = pd.read_csv(URL_GENRES, sep='\t', header=None, index_col=None, names=['msd_URI', 'genre'])

pd.merge(df_tracks, df_genres, on='msd_URI').to_csv('examples.tsv', sep='\t', index=False)