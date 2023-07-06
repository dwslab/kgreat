# syntax=docker/dockerfile:1
FROM python:3.8
RUN pip3 install pyyaml pandas scikit-learn turicreate
WORKDIR /app
ADD dataset/dm-MovieLens .
RUN wget https://files.grouplens.org/datasets/movielens/ml-20m.zip && unzip ml-20m.zip && rm ml-20m.zip
RUN wget -O dbpedia-links.tsv https://raw.githubusercontent.com/sisinflab/LODrecsys-datasets/master/Movielens1M/MappingMovielens2DBpedia-1.2.tsv
ADD shared/dm .
CMD ["python3", "."]