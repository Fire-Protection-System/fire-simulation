FROM python:3.11-slim

WORKDIR /code
RUN apt-get update && apt-get install -y build-essential gcc && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml pyproject.toml
COPY src/ src/
RUN pip install --no-cache-dir .
COPY . /code

# Ensure log directory exists for log file mount
RUN mkdir -p /var/log/fire-simulation

CMD ["python", "./main.py"]