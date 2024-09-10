FROM python:3.10-slim

WORKDIR /python-docker

COPY requirements.txt requirements.txt

RUN apt-get update && apt-get install -y ffmpeg

RUN pip install -r requirements.txt

RUN apt-get install -y git
RUN pip install "git+https://github.com/openai/whisper.git"
RUN pip install epitran[flite]

COPY . .

EXPOSE 8000

CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "${PORT:-8000}"]