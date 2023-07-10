# syntax=docker/dockerfile:1
FROM ghcr.io/dwslab/jrdf2vec
# install missing dependencies
RUN pip3 install pyyaml pandas
# add code
WORKDIR /app
ADD shared/preprocessing/embedding-rdf2vec .
# overwrite entry point
ENTRYPOINT ["python3", "."]