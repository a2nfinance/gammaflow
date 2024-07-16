FROM python:3.10.12-slim

RUN apt-get -y update && apt-get install -y git python-is-python3 ffmpeg

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip 
RUN pip install -r /app/requirements.txt

EXPOSE 5000

ENTRYPOINT ["python", "services.py"]