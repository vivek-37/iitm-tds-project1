# # Stage 1: Node.js builder
# FROM node:20-alpine AS builder

# # No need to install npx separately, it's included in Node.js
# WORKDIR /app

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (curl and Node.js)
RUN apt-get update && apt-get install -y \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g prettier@3.4.2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

# Create the /data directory and set permissions to make it writable by all users
RUN mkdir -p /data && chmod 777 /data

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]