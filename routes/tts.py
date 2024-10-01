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

router = APIRouter()
logger = logging.getLogger(__name__)

RUNNING_ON_GCLOUD = os.getenv('GOOGLE_CLOUD_PROJECT') is not None

# Initialize NLTK data
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        logger.info("NLTK punkt tokenizer already downloaded.")
    except LookupError:
        logger.info("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt', quiet=True)
        logger.info("NLTK punkt tokenizer downloaded successfully.")

# Call the function to download NLTK data
download_nltk_data()

# Initialize the TTS model
try:
    if RUNNING_ON_GCLOUD:
        tts = TTS("tts_models/en/vctk/vits")
    else:
        tts = TTS("tts_models/en/vctk/vits", gpu=torch.cuda.is_available())
    
    # Optimize model for inference
    tts.synthesizer.tts_model.eval()
    if torch.cuda.is_available():
        tts.synthesizer.tts_model.half()  # Use half-precision if GPU is available
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
audio_cache = {}

def process_sentence(sentence, speaker):
    cache_key = (sentence, speaker)
    if cache_key in audio_cache:
        return audio_cache[cache_key]

    start_time = time.time()
    wav = tts.tts(text=sentence, speaker=speaker)
    tts_time = time.time() - start_time

    byte_io = io.BytesIO()
    tts.synthesizer.save_wav(wav, byte_io)
    byte_io.seek(0)
    audio_data = byte_io.getvalue()
    
    audio_cache[cache_key] = audio_data
    
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
        # Ensure NLTK data is available
        download_nltk_data()

        # Split text into sentences
        sentences = nltk.sent_tokenize(text)
        print(f"Split text into {len(sentences)} sentences.", file=sys.stderr)

        # Process sentences in parallel
        audio_chunks = await process_sentences(sentences, speaker)

        # Encode each audio chunk to base64
        audio_base64_list = [base64.b64encode(chunk).decode() for chunk in audio_chunks]

        end_time = time.time()
        total_time = end_time - start_time
        print(f"Successfully processed text to speech. Total processing time: {total_time:.4f} seconds", file=sys.stderr)
        logger.info(f"Successfully processed text to speech. Total processing time: {total_time:.4f} seconds")
        
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