## AI BACKEND SERVER

### Intro
A server to handle AI related features for other frontend apps. The app was created by Fast API Python, hosted on GCloud. 

### Guide
Run: `docker-compose up --build`
Or run venv: 
```
    python -m venv venv
    source venv/bin/activate
    uvicorn fastapi_app:app --host 0.0.0.0 --port 8000
```

Note:
In the first run, we'll need to install the dependencies and requirements:
```
    pip install -r requirements.txt
    pip install "git+https://github.com/openai/whisper.git"
    pip install epitran[flite]
    brew install ffmpeg espeak
```
In Windows, run `venv\Scripts\activate`
```
    sudo apt-get update
    sudo apt-get install -y ffmpeg espeak-ng
```