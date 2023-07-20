# syntax=docker/dockerfile:1
FROM python:3.9
RUN pip3 install pyyaml pandas scikit-learn hnswlib
WORKDIR /app
ADD shared/dm .
ADD dataset/dm-allcapitalcountryentities .
CMD ["python3", "."]