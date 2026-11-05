FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

RUN useradd --create-home --uid 10001 robot

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config ./config
COPY examples ./examples
COPY tests ./tests

USER robot

ENTRYPOINT ["python", "examples/groot_ping.py"]
