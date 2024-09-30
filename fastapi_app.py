##fastapi_app.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
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
import base64
from typing import List
from pronunciation_trainer import getTrainer
import eng_to_ipa
from routes import dictionary
from TTS.api import TTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.include_router(dictionary.router, tags=["dictionary"])

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

# Check if we're running on Google Cloud
RUNNING_ON_GCLOUD = os.getenv('GOOGLE_CLOUD_PROJECT') is not None

# Initialize the TTS model
try:
    if RUNNING_ON_GCLOUD:
        # On Google Cloud, we'll use CPU for cost-efficiency
        tts = TTS("tts_models/en/vctk/vits")
    else:
        # Locally, we'll use GPU if available
        tts = TTS("tts_models/en/vctk/vits", gpu=torch.cuda.is_available())
except Exception as e:
    logger.error(f"Error initializing primary TTS model: {str(e)}")
    logger.info("Falling back to alternative TTS model...")
    try:
        tts = TTS("tts_models/en/ljspeech/tacotron2-DDC", gpu=torch.cuda.is_available())
    except Exception as e:
        logger.error(f"Error initializing fallback TTS model: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to initialize TTS model")

# Preload the model to memory
tts.synthesizer.tts_model.eval()

@app.post("/tts/")
async def text_to_speech(text: str = Form(...), speaker: str = Form("p243")):
    try:
        # Generate speech
        wav = tts.tts(text=text, speaker=speaker)
        
        # Convert numpy array to bytes
        byte_io = io.BytesIO()
        tts.synthesizer.save_wav(wav, byte_io)
        byte_io.seek(0)

        # Encode to base64
        audio_base64 = base64.b64encode(byte_io.getvalue()).decode()

        return JSONResponse(content={"audio": audio_base64})
    except Exception as e:
        logger.error(f"Error in text_to_speech: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TTS conversion failed: {str(e)}")

@app.on_event("startup")
async def startup_event():
    if RUNNING_ON_GCLOUD:
        # Optimize the instance on startup when running on Google Cloud
        subprocess.run(["sudo", "swapoff", "-a"])
        subprocess.run(["sudo", "sysctl", "vm.swappiness=1"])
        subprocess.run(["sudo", "sysctl", "vm.vfs_cache_pressure=50"])

@app.post("/assess_pronunciation/")
async def assess_pronunciation(
    title: str = Form(...),
    base64Audio: str = Form(...)
):
    logger.info(f"Received pronunciation assessment request for title: {title}")
    try:
        # Extract the base64 audio data
        audio_data = base64Audio.split(',')[1]
        audio_bytes = base64.b64decode(audio_data)

        # Log the size of the received audio
        logger.info(f"Received audio size: {len(audio_bytes)} bytes")

        # Convert to WAV
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="m4a")
        audio = audio.set_frame_rate(16000).set_channels(1)
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)

        # Log the size of the converted WAV
        logger.info(f"Converted WAV size: {wav_io.getbuffer().nbytes} bytes")

        # Convert to tensor
        audio_tensor, sample_rate = torchaudio.load(wav_io)
        
        logger.info(f"Audio converted to tensor: shape {audio_tensor.shape}, dtype {audio_tensor.dtype}, sample rate {sample_rate}")
        
        result = await process_audio(audio_tensor, title)
        logger.info(f"Audio processing completed. Result: {result}")
        
        return json.dumps(result)

    except Exception as e:
        logger.error(f"Error in assess_pronunciation: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error in pronunciation assessment: {str(e)}")

async def process_audio(audio_tensor: torch.Tensor, title: str):
    logger.info("Processing audio")
    try:
        # Log the input to processAudioForGivenText
        logger.info(f"Input to processAudioForGivenText: audio_tensor shape {audio_tensor.shape}, title '{title}'")
        
        result = await asyncio.to_thread(trainer.processAudioForGivenText, audio_tensor, title)
        
        # Log the output of processAudioForGivenText
        logger.info(f"Output of processAudioForGivenText: {result}")
        
        if not result['recording_transcript']:
            logger.warning("No transcription produced by the ASR model")
        
        logger.info("Audio processing completed")
        return result
    except Exception as e:
        logger.error(f"Error during audio processing: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error during audio processing: {str(e)}")

@app.post("/get_phonetic/")
async def get_phonetic(text: str = Form(...)):
    try:
        phonetic = eng_to_ipa.convert(text)
        return JSONResponse(content={"phonetic": phonetic})
    except Exception as e:
        logger.error(f"Error in get_phonetic: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error in phonetic conversion: {str(e)}")

# @app.post("/tts/")
# async def text_to_speech(text: str = Form(...), lang: str = Form("en")):
#     try:
#         tts = gTTS(text=text, lang=lang)
        
#         # Save to a BytesIO object instead of a file
#         audio_io = io.BytesIO()
#         tts.write_to_fp(audio_io)
#         audio_io.seek(0)
        
#         # Encode to base64
#         audio_base64 = base64.b64encode(audio_io.getvalue()).decode()
        
#         return JSONResponse(content={"audio": audio_base64})
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"TTS conversion failed: {str(e)}")

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

            result = whisper_model.transcribe(
                temp.name,
                language="en",  
                task="transcribe",  
            )
            results.append(
                {
                    'filename': file.filename,
                    'transcript': result["text"]
                }
            )
    return JSONResponse(content={'results': results})

@app.get("/")
async def root():
    return RedirectResponse(url="/docs")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)