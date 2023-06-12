#!/bin/sh
mkdir -p data
cd data
wget -O dbpedia-ontology.nt http://akswnc7.informatik.uni-leipzig.de/dstreitmatter/archivo/dbpedia.org/ontology--DEV/2022.10.09-192003/ontology--DEV_type=parsed.nt
wget -O dbpedia-instance-types.nt.bz2 https://downloads.dbpedia.org/repo/dbpedia/mappings/instance-types/2022.09.01/instance-types_lang=en_specific.ttl.bz2
wget -O dbpedia-mappingbased-objects.nt.bz2 https://downloads.dbpedia.org/repo/dbpedia/mappings/mappingbased-objects/2022.09.01/mappingbased-objects_lang=en.ttl.bz2
wget -O dbpedia-labels.nt.bz2 https://downloads.dbpedia.org/repo/dbpedia/generic/labels/2022.09.01/labels_lang=en.ttl.bz2