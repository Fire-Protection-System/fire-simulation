FROM python:3.11-slim

WORKDIR /code

# Install dependencies first (better caching)
COPY ./requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /code

# Run application
CMD ["python", "./main.py"]