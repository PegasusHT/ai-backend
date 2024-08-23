from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List
from fastapi.responses import JSONResponse, RedirectResponse
import whisper 
import torch
from tempfile import NamedTemporaryFile

torch.cuda.is_available()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model = whisper.load_model('base', device=DEVICE)

app = FastAPI()

@app.post("/whisper/")
async def handler(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    results = []

    for file in files:
        with NamedTemporaryFile(delete=False) as temp:
            with open(temp.name, 'wb') as temp_file:
                temp_file.write(file.file.read())

            result = model.transcribe(temp.name)
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