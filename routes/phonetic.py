from fastapi import APIRouter, HTTPException, Form
from fastapi.responses import JSONResponse
import eng_to_ipa
import logging
import traceback

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/get_phonetic/")
async def get_phonetic(text: str = Form(...)):
    try:
        phonetic = eng_to_ipa.convert(text)
        return JSONResponse(content={"phonetic": phonetic})
    except Exception as e:
        logger.error(f"Error in get_phonetic: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error in phonetic conversion: {str(e)}")