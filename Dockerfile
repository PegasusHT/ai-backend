# FROM --platform=linux/amd64 python:3.10-slim
FROM python:3.10-slim

WORKDIR /python-docker

COPY requirements.txt requirements.txt

RUN apt-get update && \
    apt-get install -y ffmpeg git espeak-ng libsndfile1 && \
    pip install -r requirements.txt && \
    pip install "git+https://github.com/openai/whisper.git" && \
    pip install epitran[flite] && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY . .

# Print directory contents for debugging
RUN echo "Directory contents:" && ls -la

EXPOSE 8000

# Print environment variables and attempt to start the app
CMD echo "PORT: $PORT" && \
    echo "Directory contents:" && ls -la && \
    uvicorn fastapi_app:app --host 0.0.0.0 --port $PORT