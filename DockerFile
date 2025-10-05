# Dockerfile Name Chaned to DockerFile
FROM python:3.9-slim

WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the training source code
COPY ./src .

# The entrypoint to run our training script
ENTRYPOINT ["python", "train.py"]