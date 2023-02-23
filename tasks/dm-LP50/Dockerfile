# syntax=docker/dockerfile:1
FROM python:3.9
WORKDIR /app
ADD tasks/dm-LP50 .
ADD shared/dm .

ENV KGREAT_DATASET='{"name": "LP50", "format": "documentSimilarity", "entity_keys": ["DBpedia16_URI"], "entity_file": "entities.json", "docsim_file": "docsim.csv"}'

RUN pip3 install pyyaml pandas scikit-learn

CMD ["python3", "."]