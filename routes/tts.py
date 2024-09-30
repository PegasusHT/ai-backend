from fastapi import APIRouter, HTTPException, Form
from fastapi.responses import JSONResponse
from TTS.api import TTS
import torch
import io
import base64
import os
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

RUNNING_ON_GCLOUD = os.getenv('GOOGLE_CLOUD_PROJECT') is not None

# Initialize the TTS model
try:
    if RUNNING_ON_GCLOUD:
        tts = TTS("tts_models/en/vctk/vits")
    else:
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

@router.post("/tts/")
async def text_to_speech(text: str = Form(...), speaker: str = Form("p240")):
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