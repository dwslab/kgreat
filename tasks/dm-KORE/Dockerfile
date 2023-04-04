# syntax=docker/dockerfile:1
FROM python:3.9
WORKDIR /app
ADD tasks/dm-KORE .
ADD shared/dm .

RUN pip3 install pyyaml pandas scikit-learn

CMD ["python3", "."]