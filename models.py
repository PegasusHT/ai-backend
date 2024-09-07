import torch
import torch.nn as nn

def getASRModel():
    model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                           model='silero_stt',
                                           language='en',
                                           device=torch.device('cpu'))
    return model, decoder

def getTTSModel():
    speaker = 'lj_16khz'  # 16 kHz
    model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                              model='silero_tts',
                              language='en',
                              speaker=speaker)
    return model