FROM --platform=linux/amd64 python:3.10-slim

WORKDIR /app

COPY requirements.txt requirements.txt

RUN apt-get update && \
    apt-get install -y ffmpeg git espeak-ng libsndfile1 && \
    pip install -r requirements.txt && \
    pip install "git+https://github.com/openai/whisper.git" && \
    pip install epitran[flite] && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /root/nltk_data && \
    python -m nltk.downloader -d /root/nltk_data punkt
    
COPY . .

EXPOSE 8000

CMD uvicorn fastapi_app:app --host 0.0.0.0 --port $PORT