# syntax=docker/dockerfile:1
FROM python:3.8
RUN pip3 install pyyaml pandas scikit-learn turicreate
WORKDIR /app
ADD dataset/dm-LastFm .
RUN wget https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip && unzip hetrec2011-lastfm-2k.zip && rm hetrec2011-lastfm-2k.zip
RUN wget -O dbpedia-links.tsv https://raw.githubusercontent.com/sisinflab/LODrecsys-datasets/master/LastFM/MappingLastfm2DBpedia-1.2.tsv
ADD shared/dm .
CMD ["python3", "."]