#!/bin/sh
mkdir -p data && cd data
wget https://data.dws.informatik.uni-mannheim.de/kgreat/wikidata_labels.nt
wget https://data.dws.informatik.uni-mannheim.de/kgreat/wikidata_sameas.nt
cd .. && mkdir -p embeddings && cd embeddings
wget -O TransE_l1.tsv.gz https://dl.fbaipublicfiles.com/torchbiggraph/wikidata_translation_v1.tsv.gz && gzip -d TransE_l1.tsv.gz