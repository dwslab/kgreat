# syntax=docker/dockerfile:1
FROM python:3.9
RUN pip3 install pyyaml pandas scikit-learn
WORKDIR /app
ADD shared/dm .
ADD dataset/dm-comiccharacters .
CMD ["python3", "."]