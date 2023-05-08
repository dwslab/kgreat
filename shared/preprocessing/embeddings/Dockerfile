# syntax=docker/dockerfile:1
FROM python:3.8
WORKDIR /app
ADD shared/preprocessing/embeddings .

RUN pip3 install pyyaml pandas torch
RUN pip3 install dgl==0.4.3
RUN pip3 install dglke

ENV DGLBACKEND=pytorch

CMD ["python3", "."]