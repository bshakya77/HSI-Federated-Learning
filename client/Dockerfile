FROM python:3.10-slim

WORKDIR /app

COPY client/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY shared/ /app/shared/
COPY client/train.py .

RUN mkdir -p /app/outputs /app/logs

ENV PYTHONUNBUFFERED=1

CMD ["python", "train.py"]
