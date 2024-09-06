from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware  
from typing import List
from fastapi.responses import JSONResponse, RedirectResponse, FileResponse
import whisper
import torch
from tempfile import NamedTemporaryFile
from gtts import gTTS
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "https://interview-ace-ai.vercel.app",
        'https://ai-backend-378206958409.us-east1.run.app'
    ],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

def cleanup(path: str):
    if os.path.exists(path):
        os.unlink(path)

@app.post("/tts/")
async def text_to_speech(background_tasks: BackgroundTasks, text: str = Form(...), lang: str = Form("en")):
    try:
        tts = gTTS(text=text, lang=lang)
        
        with NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            temp_audio_path = temp_audio.name
            tts.save(temp_audio_path)
        
        background_tasks.add_task(cleanup, temp_audio_path)
        
        return FileResponse(temp_audio_path, media_type="audio/mpeg", filename="tts_output.mp3")
    except Exception as e:
        if 'temp_audio_path' in locals() and os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)
        raise HTTPException(status_code=500, detail=f"TTS conversion failed: {str(e)}")

def get_whisper_model():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    return whisper.load_model('base', device=DEVICE)

@app.post("/whisper/")
async def handler(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    logger.info("Starting transcription")
    results = []
    model = get_whisper_model()

    for file in files:
        with NamedTemporaryFile(delete=False) as temp:
            with open(temp.name, 'wb') as temp_file:
                temp_file.write(file.file.read())

            result = model.transcribe(temp.name)
            results.append(
                {
                    'filename': file.filename,
                    'transcript': result["text"]
                }
            )
    return JSONResponse(content={'results': results})

@app.get("/", response_class=RedirectResponse)
async def redirect_to_docs():
    return "/docs"