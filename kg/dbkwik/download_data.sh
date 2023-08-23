#!/bin/sh
mkdir -p data
cd data
wget https://figshare.com/ndownloader/files/36488608
tar -xzvf 36488608
#mkdir -p files_to_index
#cp labels.ttl article-categories.ttl short-abstracts.ttl template-type.ttl template-type-definitions.ttl infobox-template-type.ttl infobox-template-type-definitions.ttl infobox-properties.ttl files_to_index/
