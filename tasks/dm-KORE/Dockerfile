# syntax=docker/dockerfile:1
FROM python:3.9
WORKDIR /app
ADD tasks/dm-KORE .
ADD shared/dm .

ENV KGREAT_DATASET='{"name": "KORE", "format": "entityRelatedness", "data_file": "examples.tsv", "entity_keys": ["DBpedia16_URI"]}'

RUN pip3 install pyyaml pandas scikit-learn

CMD ["python3", "."]