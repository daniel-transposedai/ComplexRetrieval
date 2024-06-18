FROM python:3.11-slim

WORKDIR /code

COPY ./app/ ./app

COPY ./pyproject.toml ./README.md ./requirements.txt ./

COPY ./utils ./utils

RUN pip install -r requirements.txt

EXPOSE 8080

CMD streamlit run code/app/app.py
