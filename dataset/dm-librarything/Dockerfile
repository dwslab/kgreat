# syntax=docker/dockerfile:1
FROM python:3.8
RUN pip3 install pyyaml pandas scikit-learn turicreate
WORKDIR /app
ADD dataset/dm-librarything .
RUN wget https://datarepo.eng.ucsd.edu/mcauley_group/data/librarything/lthing_data.tar.gz && tar -xzvf lthing_data.tar.gz && rm lthing_data.tar.gz
RUN wget -O dbpedia-links.tsv https://raw.githubusercontent.com/sisinflab/LODrecsys-datasets/master/LibraryThing/MappingLibrarything2DBpedia-1.2.tsv
ADD shared/dm .
CMD ["python3", "."]