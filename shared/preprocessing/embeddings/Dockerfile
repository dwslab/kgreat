# syntax=docker/dockerfile:1
FROM python:3.9
WORKDIR /app
ADD shared/preprocessing/embeddings .

RUN pip3 install pyyaml torch
RUN git clone https://github.com/facebookresearch/PyTorch-BigGraph.git pbg
RUN (cd pbg; pip3 install .)

CMD ["python3", "."]