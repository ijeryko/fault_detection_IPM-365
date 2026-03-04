FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir --default-timeout=120 --retries=10 -r /app/requirements.txt

COPY . /app

# Ensure services is importable (package)
RUN test -f /app/services/__init__.py || echo "# package" > /app/services/__init__.py

# HARD PROOF during build: list services directory
RUN echo "=== /app/services ===" && ls -la /app/services

EXPOSE 5000
CMD ["python", "/app/main.py"]