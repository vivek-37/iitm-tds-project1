FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (including Node.js and npm for npx)
RUN apt-get update && apt-get install -y nodejs npm && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

# Create the /data directory and set permissions to make it writable by all users
RUN mkdir -p /data && chmod 777 /data

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]