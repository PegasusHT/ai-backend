from fastapi import APIRouter, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from typing import List
import whisper
from tempfile import NamedTemporaryFile
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

DEVICE = 'cpu'
whisper_model = whisper.load_model('base', device=DEVICE)

@router.post("/whisper/")
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