# # Stage 1: Node.js builder
# FROM node:20-alpine AS builder

# # No need to install npx separately, it's included in Node.js
# WORKDIR /app

FROM python:3.11-slim

WORKDIR /app

# RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

# Copy npx and Node.js binary from the builder stage
# COPY --from=builder /usr/local/bin/node /usr/local/bin/node
# COPY --from=builder /usr/local/bin/npm /usr/local/bin/npm
# COPY --from=builder /usr/local/bin/npx /usr/local/bin/npx
# COPY --from=builder /usr/local/lib/node_modules /usr/local/lib/node_modules

# Create the /data directory and set permissions to make it writable by all users
RUN mkdir -p /data && chmod 777 /data

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]