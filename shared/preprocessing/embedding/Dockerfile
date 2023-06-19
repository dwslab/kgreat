# syntax=docker/dockerfile:1
FROM nvidia/cuda:11.6.2-base-ubuntu20.04
# install python 3.8
RUN apt update && apt install software-properties-common -y && add-apt-repository ppa:deadsnakes/ppa && apt install python3.8 python3-pip -y
# install dependencies
RUN pip3 install pyyaml pandas torch
RUN pip3 install dgl==0.4.3
RUN pip3 install dglke
# add code
WORKDIR /app
ADD shared/preprocessing/embedding .
# make configurations for dgl-ke
ENV DGLBACKEND=pytorch
CMD ["python3", "."]