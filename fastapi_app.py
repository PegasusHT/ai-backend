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
import torchaudio
import json
import traceback
import io
from pydub import AudioSegment
import asyncio

from pronunciation_trainer import PronunciationTrainer, getTrainer

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

DEVICE = 'cpu'
whisper_model = whisper.load_model('base', device=DEVICE)

trainer = getTrainer()

def cleanup(path: str):
    if os.path.exists(path):
        os.unlink(path)

@app.post("/assess_pronunciation/")
async def assess_pronunciation(
    title: str = Form(...),
    audio: UploadFile = File(...)
):
    print(f"Received pronunciation assessment request for title: {title}")
    try:
        audio_content = await audio.read()
        print(f"Audio file: {audio.filename}, Content-Type: {audio.content_type}, Size: {len(audio_content)} bytes")
        
        audio_tensor = await convert_audio_to_tensor(audio_content)
        
        result = await process_audio(audio_tensor, title)
        
        return JSONResponse(content=result)

    except Exception as e:
        print(f"Unexpected error in assess_pronunciation: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Unexpected error in pronunciation assessment: {str(e)}")

async def convert_audio_to_tensor(audio_content: bytes) -> torch.Tensor:
    print("Converting audio to WAV format...")
    audio_segment = AudioSegment.from_file(io.BytesIO(audio_content), format="m4a")
    wav_io = io.BytesIO()
    audio_segment.export(wav_io, format="wav")
    wav_io.seek(0)
    
    print("Converting to torch tensor...")
    audio_tensor, _ = torchaudio.load(wav_io)
    return audio_tensor

async def process_audio(audio_tensor: torch.Tensor, title: str):
    print("Processing audio...")
    try:
        result = await asyncio.to_thread(trainer.processAudioForGivenText, audio_tensor, title)
        print("Audio processing completed")
        return result
    except Exception as e:
        print(f"Error during audio processing: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error during audio processing: {str(e)}")

@app.post("/tts/")
async def text_to_speech(background_tasks: BackgroundTasks, text: str = Form(...)):
    try:
        tts = gTTS(text=text, lang="en")
        
        with NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            temp_audio_path = temp_audio.name
            tts.save(temp_audio_path)
        
        background_tasks.add_task(cleanup, temp_audio_path)
        
        return FileResponse(temp_audio_path, media_type="audio/mpeg", filename="tts_output.mp3")
    except Exception as e:
        if 'temp_audio_path' in locals() and os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)
        raise HTTPException(status_code=500, detail=f"TTS conversion failed: {str(e)}")

@app.post("/whisper/")
async def handler(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    logger.info("Starting transcription")
    results = []

    for file in files:
        with NamedTemporaryFile(delete=False) as temp:
            with open(temp.name, 'wb') as temp_file:
                temp_file.write(file.file.read())

            result = whisper_model.transcribe(temp.name)
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

@app.get("/test")
async def test_endpoint():
    return {"message": "Test endpoint working"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)