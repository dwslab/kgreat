# syntax=docker/dockerfile:1
FROM python:3.9
WORKDIR /app
ADD tasks/dm-MetacriticAlbums .
ADD shared/dm .

ENV KGREAT_DATASET='{"name": "MetacriticAlbums", "format": "tsv", "data_file": "examples.tsv", "entity_keys": ["Wikidata_URI", "DBpedia15_URI", "DBpedia16_URI"], "label": "label"}'

RUN pip3 install pyyaml pandas scikit-learn

CMD ["python3", "."]