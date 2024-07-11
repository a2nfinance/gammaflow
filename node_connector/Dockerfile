# Build an image that can serve mlflow models.
FROM python:3.10.12-slim

RUN apt-get -y update && apt-get install -y git python-is-python3

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip 
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

EXPOSE 5000

ENTRYPOINT ["python", "main.py"]