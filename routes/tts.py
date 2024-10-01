from fastapi import APIRouter, HTTPException, Form
from fastapi.responses import JSONResponse
from TTS.api import TTS
import torch
import io
import base64
import os
import logging
import re

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

def split_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text)

def text_to_audio(text, speaker):
    wav = tts.tts(text=text, speaker=speaker)
    byte_io = io.BytesIO()
    tts.synthesizer.save_wav(wav, byte_io)
    byte_io.seek(0)
    return base64.b64encode(byte_io.getvalue()).decode()

@router.post("/quick")
async def quick_text_to_speech(text: str = Form(...), speaker: str = Form("p240")):
    try:
        sentences = split_sentences(text)
        first_two_sentences = ' '.join(sentences[:2])
        audio_base64 = text_to_audio(first_two_sentences, speaker)
        return JSONResponse(content={
            "audio": audio_base64, 
            "is_partial": True,
        })
    except Exception as e:
        logger.error(f"Error in quick_text_to_speech: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Quick TTS conversion failed: {str(e)}")

@router.post("/full")
async def full_text_to_speech(text: str = Form(...), speaker: str = Form("p240")):
    try:
        sentences = split_sentences(text)
        remaining_sentences = ' '.join(sentences[2:])  # Process all sentences after the first two
        audio_base64 = text_to_audio(remaining_sentences, speaker)
        return JSONResponse(content={
            "audio": audio_base64, 
            "is_partial": False,
        })
    except Exception as e:
        logger.error(f"Error in full_text_to_speech: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Full TTS conversion failed: {str(e)}")