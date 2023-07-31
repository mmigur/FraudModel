FROM python:latest

RUN mkdir app
COPY ./ /app
COPY ./requirements.txt /app/
RUN pip install --upgrade pip && pip install -r /app/requirements.txt
RUN python main.py