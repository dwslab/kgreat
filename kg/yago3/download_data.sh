#!/bin/sh
mkdir -p data && cd data
wget https://yago-knowledge.org/data/yago3/yago-3.0.2-turtle-simple.7z && 7z x yago-3.0.2-turtle-simple.7z && rm yago-3.0.2-turtle-simple.7z