# syntax=docker/dockerfile:1
FROM python:3.9
RUN pip3 install pyyaml pandas hnswlib
WORKDIR /app
ADD shared/preprocessing/embedding-speedup .
CMD ["python3", "."]