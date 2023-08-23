#!/bin/bash
mkdir -p data && cd data

FILES=('yago-wd-class.nt.gz' 'yago-wd-schema.nt.gz' 'yago-wd-full-types.nt.gz' 'yago-wd-facts.nt.gz' 'yago-wd-labels.nt.gz' 'yago-wd-sameAs.nt.gz')
for file in "${FILES[@]}"
do
  wget https://yago-knowledge.org/data/yago4/en/2020-02-24/$file && gunzip $file
done