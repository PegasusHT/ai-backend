from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
import logging
import os
from routes import tts, asr, pronunciation, phonetic, dictionary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(tts.router, prefix="/tts", tags=["text-to-speech"])
app.include_router(asr.router, tags=["speech-recognition"])
app.include_router(pronunciation.router, tags=["pronunciation"])
app.include_router(phonetic.router, tags=["phonetic"])
app.include_router(dictionary.router, tags=["dictionary"])

# Check if we're running on Google Cloud
RUNNING_ON_GCLOUD = os.getenv('GOOGLE_CLOUD_PROJECT') is not None

@app.on_event("startup")
async def startup_event():
    if RUNNING_ON_GCLOUD:
        # Optimize the instance on startup when running on Google Cloud
        import subprocess
        subprocess.run(["sudo", "swapoff", "-a"])
        subprocess.run(["sudo", "sysctl", "vm.swappiness=1"])
        subprocess.run(["sudo", "sysctl", "vm.vfs_cache_pressure=50"])
    # No need to download NLTK data here since it's already downloaded in Dockerfile

@app.get("/")
async def root():
    return RedirectResponse(url="/docs")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
