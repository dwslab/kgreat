#!/bin/sh
mkdir -p data
cd data
wget http://downloads.dbpedia.org/2016-10/dbpedia_2016-10.nt
wget http://downloads.dbpedia.org/2016-10/core-i18n/en/instance_types_en.ttl.bz2
wget http://downloads.dbpedia.org/2016-10/core-i18n/en/mappingbased_objects_en.ttl.bz2