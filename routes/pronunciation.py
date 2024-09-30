from fastapi import APIRouter, HTTPException, Form
from fastapi.responses import JSONResponse
import logging
import base64
import io
from pydub import AudioSegment
import torchaudio
import json
import traceback
import asyncio
from pronunciation_trainer import getTrainer

router = APIRouter()
logger = logging.getLogger(__name__)

trainer = getTrainer()

@router.post("/assess_pronunciation/")
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
        
        return JSONResponse(content=json.dumps(result))

    except Exception as e:
        logger.error(f"Error in assess_pronunciation: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error in pronunciation assessment: {str(e)}")

async def process_audio(audio_tensor, title: str):
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