# syntax=docker/dockerfile:1
FROM python:3.9
RUN pip3 install pyyaml pandas rapidfuzz
WORKDIR /app
ADD shared/mapping/label .
CMD ["python3", "."]