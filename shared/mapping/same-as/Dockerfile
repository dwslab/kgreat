# syntax=docker/dockerfile:1
FROM python:3.9
RUN pip3 install pyyaml pandas
WORKDIR /app
ADD shared/mapping/same-as .
CMD ["python3", "."]