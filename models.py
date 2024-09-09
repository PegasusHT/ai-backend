import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

def getASRModel():
    logger.info("Loading ASR model...")
    model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                           model='silero_stt',
                                           language='en',
                                           device=torch.device('cpu'))
    logger.info("ASR model loaded successfully")
    return model, decoder

def getTTSModel():
    speaker = 'lj_16khz'  # 16 kHz
    model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                              model='silero_tts',
                              language='en',
                              speaker=speaker)
    return model