FROM python:3.11-slim

WORKDIR /code

COPY ./app/ ./app

COPY ./LICENSE.md ./full.yaml ./README.md ./requirements_m3.txt ./

COPY ./util ./util

RUN pip install -r requirements_m3.txt

# Exposing for dashboard webserver, if running via docker exec.
EXPOSE 61009

CMD python -m app.eval
