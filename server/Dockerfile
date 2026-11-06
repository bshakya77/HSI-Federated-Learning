FROM python:3.10-slim

WORKDIR /app

COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY shared/ /app/shared/
COPY server/app.py .

RUN mkdir -p /app/outputs /app/logs

ENV PYTHONUNBUFFERED=1
EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import socket; s=socket.create_connection(('localhost',8080),2); s.close()"

CMD ["python", "app.py"]
