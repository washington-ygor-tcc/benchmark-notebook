FROM python:3-slim

WORKDIR /app

RUN apt-get update &&\
    apt-get install git -y

COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY ./notebooks /app/notebooks

EXPOSE 4000
