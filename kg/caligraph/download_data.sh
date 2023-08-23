#!/bin/sh
mkdir -p data
cd data
wget https://zenodo.org/record/8068322/files/caligraph-ontology.nt.bz2
wget https://zenodo.org/record/8068322/files/caligraph-instances_types.nt.bz2
wget https://zenodo.org/record/8068322/files/caligraph-instances_relations.nt.bz2
wget https://zenodo.org/record/8068322/files/caligraph-instances_dbpedia-mapping.nt.bz2
wget https://zenodo.org/record/8068322/files/caligraph-instances_labels.nt.bz2