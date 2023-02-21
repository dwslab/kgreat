# syntax=docker/dockerfile:1
FROM python:3.9
WORKDIR /app
ADD tasks/dm-Cities2000AndCountries .
ADD shared/dm .

ENV KGREAT_DATASET='{"name": "Cities2000AndCountries", "format": "tsv", "data_file": "examples.tsv", "entity_keys": ["DBpedia16_URI"], "label": "cluster"}'

RUN pip3 install pyyaml pandas scikit-learn

CMD ["python3", "."]