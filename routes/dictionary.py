##path: routes/dictionary.py
from fastapi import APIRouter, HTTPException
import requests
from pydantic import BaseModel

router = APIRouter()

DICTIONARY_API_URL = "https://api.dictionaryapi.dev/api/v2/entries/en/"

class DictionaryResponse(BaseModel):
    word: str
    phonetic: str
    meanings: list
    audio_url: str = None

@router.get("/dictionary/{word}", response_model=DictionaryResponse)
async def get_word_definition(word: str):
    try:
        response = requests.get(f"{DICTIONARY_API_URL}{word}")
        if response.status_code == 200:
            data = response.json()[0]
            return DictionaryResponse(
                word=data['word'],
                phonetic=data.get('phonetic', ''),
                meanings=[
                    {
                        'part_of_speech': meaning['partOfSpeech'],
                        'definitions': [
                            {
                                'definition': d['definition'],
                                'example': d.get('example', '')
                            } for d in meaning['definitions'][:3]  # Limit to 3 definitions
                        ]
                    } for meaning in data['meanings']
                ],
                audio_url=next((p['audio'] for p in data.get('phonetics', []) if p.get('audio')), None)
            )
        else:
            raise HTTPException(status_code=404, detail="Word not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))