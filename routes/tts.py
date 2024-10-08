from fastapi import APIRouter, HTTPException, Form
from fastapi.responses import JSONResponse
from TTS.api import TTS
import torch
import io
import base64
import os
import logging
import asyncio
import nltk
from concurrent.futures import ThreadPoolExecutor
import time
import sys
from functools import lru_cache
from collections import OrderedDict

router = APIRouter()
logger = logging.getLogger(__name__)

RUNNING_ON_GCLOUD = os.getenv('GOOGLE_CLOUD_PROJECT') is not None

# Initialize NLTK data
nltk.download('punkt', quiet=True)

# Initialize the TTS model
try:
    if RUNNING_ON_GCLOUD:
        tts = TTS("tts_models/en/vctk/vits")
    else:
        tts = TTS("tts_models/en/vctk/vits", gpu=torch.cuda.is_available())
    
    # Optimize model for inference
    tts.synthesizer.tts_model.eval()
    if torch.cuda.is_available():
        tts.synthesizer.tts_model.half()
        tts.synthesizer.tts_model.cuda()
except Exception as e:
    logger.error(f"Error initializing primary TTS model: {str(e)}")
    logger.info("Falling back to alternative TTS model...")
    try:
        tts = TTS("tts_models/en/ljspeech/tacotron2-DDC", gpu=torch.cuda.is_available())
    except Exception as e:
        logger.error(f"Error initializing fallback TTS model: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to initialize TTS model")

# Cache for storing generated audio
class LRUCache:
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

audio_cache = LRUCache(200)

def process_sentence(sentence, speaker):
    cache_key = (sentence, speaker)
    cached_audio = audio_cache.get(cache_key)
    if cached_audio:
        return cached_audio

    start_time = time.time()
    wav = tts.tts(text=sentence, speaker=speaker)
    tts_time = time.time() - start_time

    byte_io = io.BytesIO()
    tts.synthesizer.save_wav(wav, byte_io)
    byte_io.seek(0)
    audio_data = byte_io.getvalue()
    
    audio_cache.put(cache_key, audio_data)
    
    process_time = time.time() - start_time
    print(f"Processed sentence: '{sentence[:30]}...'. TTS time: {tts_time:.4f}s, Total time: {process_time:.4f}s", file=sys.stderr)
    
    return audio_data

async def process_sentences(sentences, speaker):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        audio_chunks = await asyncio.gather(
            *[loop.run_in_executor(executor, process_sentence, sentence, speaker) for sentence in sentences]
        )
    return audio_chunks

@router.post("/")
async def text_to_speech(text: str = Form(...), speaker: str = Form("p240")):
    start_time = time.time()
    try:
        print(f"Received text: {text[:100]}...")
        
        # Split text into sentences
        sentences = nltk.sent_tokenize(text)
        print(f"Split into {len(sentences)} sentences.", file=sys.stderr)

        # Process sentences in parallel
        audio_chunks = await process_sentences(sentences, speaker)

        # Encode each audio chunk to base64
        audio_base64_list = [base64.b64encode(chunk).decode() for chunk in audio_chunks]

        end_time = time.time()
        total_time = end_time - start_time
        print(f"Successfully processed text to speech. Total processing time: {total_time:.4f} seconds", file=sys.stderr)
        
        return JSONResponse(content={
            "audio": ",".join(audio_base64_list),
            "processing_time": total_time
        })
    except Exception as e:
        end_time = time.time()
        total_time = end_time - start_time
        error_message = f"Error in text_to_speech: {str(e)}. Total processing time: {total_time:.4f} seconds"
        print(error_message, file=sys.stderr)
        logger.error(error_message, exc_info=True)
        raise HTTPException(status_code=500, detail=f"TTS conversion failed: {str(e)}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')