#!/bin/sh
mkdir -p data && cd data
wget https://data.dws.informatik.uni-mannheim.de/kgreat/wikidata_labels.nt
wget https://data.dws.informatik.uni-mannheim.de/kgreat/wikidata_sameas.nt
wget -O wikidata_sameas_dbpedia.nt.bz2 https://downloads.dbpedia.org/repo/dbpedia/wikidata/sameas-all-wikis/2022.12.01/sameas-all-wikis.ttl.bz2 && bunzip2 wikidata_sameas_dbpedia.nt.bz2
sed -i 's|wikidata.dbpedia.org/resource|wikidata.org/entity|g' wikidata_sameas_dbpedia.nt
cd .. && mkdir -p embeddings && cd embeddings
wget -O TransE_l1.tsv.gz https://dl.fbaipublicfiles.com/torchbiggraph/wikidata_translation_v1.tsv.gz && gzip -d TransE_l1.tsv.gz