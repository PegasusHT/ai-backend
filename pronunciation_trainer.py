import torch
import numpy as np
from string import punctuation
import time
import eng_to_ipa
from models import getASRModel
import word_matching
import word_metrics

def getTrainer():
    device = torch.device('cpu')
    
    model, decoder = getASRModel()
    model = model.to(device)
    model.eval()
    
    trainer = PronunciationTrainer(model, decoder)
    return trainer

class PronunciationTrainer:
    def __init__(self, asr_model, decoder):
        self.asr_model = asr_model
        self.decoder = decoder
        self.sampling_rate = 16000
        self.categories_thresholds = np.array([80, 60, 59])

    def processAudioForGivenText(self, recordedAudio: torch.Tensor, real_text: str):
        try:
            start = time.time()
            recording_transcript = self.getAudioTranscript(recordedAudio)
            print(f'Time for ASR to transcribe audio: {time.time() - start}')
            print(f'ASR Transcript: "{recording_transcript}"')

            if not recording_transcript:
                print("ASR produced an empty transcript")
                return self.create_empty_result(real_text)

            start = time.time()
            real_and_transcribed_words, real_and_transcribed_words_ipa, mapped_words_indices = self.matchSampleAndRecordedWords(
                real_text, recording_transcript)
            print(f'Time for matching transcripts: {time.time() - start}')

            pronunciation_accuracy, current_words_pronunciation_accuracy = self.getPronunciationAccuracy(
                real_and_transcribed_words_ipa)

            pronunciation_categories = self.getWordsPronunciationCategory(
                current_words_pronunciation_accuracy)

            # Generate phonetic transcriptions for real and recorded words
            real_words_phonetic = [eng_to_ipa.convert(word) for word in real_text.split()]
            recorded_words_phonetic = [eng_to_ipa.convert(word) for word in recording_transcript.split()]

            result = {
                'recording_transcript': recording_transcript,
                'real_and_transcribed_words': real_and_transcribed_words,
                'real_and_transcribed_words_ipa': real_and_transcribed_words_ipa,
                'pronunciation_accuracy': float(pronunciation_accuracy),
                'current_words_pronunciation_accuracy': [float(acc) for acc in current_words_pronunciation_accuracy],
                'pronunciation_categories': [int(cat) for cat in pronunciation_categories],
                'real_words_phonetic': real_words_phonetic,
                'recorded_words_phonetic': recorded_words_phonetic
            }

            return self.convert_to_serializable(result)
        except Exception as e:
            print(f"Error in processAudioForGivenText: {str(e)}")
            return self.create_empty_result(real_text)

    def create_empty_result(self, real_text):
        words = real_text.split()
        return {
            'recording_transcript': '',
            'real_and_transcribed_words': [(word, '-') for word in words],
            'real_and_transcribed_words_ipa': [(eng_to_ipa.convert(word), '-') for word in words],
            'pronunciation_accuracy': 0.0,
            'current_words_pronunciation_accuracy': [0.0] * len(words),
            'pronunciation_categories': [2] * len(words),
            'real_words_phonetic': [eng_to_ipa.convert(word) for word in words],
            'recorded_words_phonetic': []
        }

    def convert_to_serializable(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_serializable(item) for item in obj]
        else:
            return obj

    def getAudioTranscript(self, recordedAudio: torch.Tensor):
        recordedAudio = self.preprocessAudio(recordedAudio)
        if recordedAudio.dim() == 2 and recordedAudio.size(0) > 1:
            recordedAudio = recordedAudio.mean(dim=0, keepdim=True)
        elif recordedAudio.dim() == 1:
            recordedAudio = recordedAudio.unsqueeze(0)
        
        print(f"Preprocessed audio shape: {recordedAudio.shape}")
        
        with torch.no_grad():
            emission = self.asr_model(recordedAudio)
        
        print(f"ASR model emission shape: {emission.shape}")
        
        transcript = self.decoder(emission[0])
        return transcript

    def matchSampleAndRecordedWords(self, real_text, recorded_transcript):
        words_estimated = recorded_transcript.split()
        words_real = real_text.split()

        mapped_words, mapped_words_indices = word_matching.get_best_mapped_words(words_estimated, words_real)

        real_and_transcribed_words = []
        real_and_transcribed_words_ipa = []
        for word_idx in range(len(words_real)):
            if word_idx >= len(mapped_words):
                mapped_words.append('-')
            real_word = words_real[word_idx]
            mapped_word = mapped_words[word_idx]
            real_and_transcribed_words.append((real_word, mapped_word))
            
            real_phonemes = eng_to_ipa.convert(real_word)
            mapped_phonemes = eng_to_ipa.convert(mapped_word) if mapped_word != '-' else '-'
            
            real_and_transcribed_words_ipa.append((real_phonemes, mapped_phonemes))
        
        return real_and_transcribed_words, real_and_transcribed_words_ipa, mapped_words_indices

    def getPronunciationAccuracy(self, real_and_transcribed_words_ipa):
        total_mismatches = 0.
        number_of_phonemes = 0.
        current_words_pronunciation_accuracy = []
        for real_phonemes, transcribed_phonemes in real_and_transcribed_words_ipa:
            real_without_punctuation = self.removePunctuation(real_phonemes).lower()
            transcribed_without_punctuation = self.removePunctuation(transcribed_phonemes).lower()
            
            number_of_word_mismatches = word_metrics.edit_distance_python(
                real_without_punctuation, transcribed_without_punctuation)
            total_mismatches += number_of_word_mismatches
            number_of_phonemes_in_word = len(real_without_punctuation)
            number_of_phonemes += number_of_phonemes_in_word

            word_accuracy = 100 * (number_of_phonemes_in_word - number_of_word_mismatches) / number_of_phonemes_in_word if number_of_phonemes_in_word > 0 else 0
            current_words_pronunciation_accuracy.append(word_accuracy)

        if number_of_phonemes == 0:
            return 0, []

        percentage_of_correct_pronunciations = 100 * (number_of_phonemes - total_mismatches) / number_of_phonemes

        return percentage_of_correct_pronunciations, current_words_pronunciation_accuracy

    def getWordsPronunciationCategory(self, accuracies):
        return [self.getPronunciationCategoryFromAccuracy(accuracy) for accuracy in accuracies]

    def getPronunciationCategoryFromAccuracy(self, accuracy):
        return np.argmin(abs(self.categories_thresholds - accuracy))

    def removePunctuation(self, word: str):
        return ''.join([char for char in word if char not in punctuation])

    def preprocessAudio(self, audio: torch.Tensor):
        audio = audio - torch.mean(audio)
        audio = audio / torch.max(torch.abs(audio))
        return audio