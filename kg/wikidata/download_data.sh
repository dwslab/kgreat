#!/bin/sh
mkdir -p embeddings && cd embeddings
wget -O TransE_l1.tsv.gz https://dl.fbaipublicfiles.com/torchbiggraph/wikidata_translation_v1.tsv.gz && gzip -d TransE_l1.tsv.gz