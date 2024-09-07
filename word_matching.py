import numpy as np
from dtwalign import dtw_from_distance_matrix
from word_metrics import edit_distance_python

def get_word_distance_matrix(words_estimated: list, words_real: list) -> np.array:
    number_of_real_words = len(words_real)
    number_of_estimated_words = len(words_estimated)

    word_distance_matrix = np.zeros((number_of_estimated_words + 1, number_of_real_words))
    for idx_estimated in range(number_of_estimated_words):
        for idx_real in range(number_of_real_words):
            word_distance_matrix[idx_estimated, idx_real] = edit_distance_python(
                words_estimated[idx_estimated], words_real[idx_real])

    for idx_real in range(number_of_real_words):
        word_distance_matrix[number_of_estimated_words, idx_real] = len(words_real[idx_real])
    
    return word_distance_matrix

def get_best_mapped_words(words_estimated: list, words_real: list) -> tuple:
    word_distance_matrix = get_word_distance_matrix(words_estimated, words_real)
    
    mapped_indices = dtw_from_distance_matrix(word_distance_matrix).path[:len(words_estimated), 1]

    mapped_words = []
    mapped_words_indices = []
    for word_idx in range(len(words_real)):
        positions = np.where(mapped_indices == word_idx)[0]
        if len(positions) == 0:
            mapped_words.append('-')
            mapped_words_indices.append(-1)
        else:
            best_pos = positions[0]
            mapped_words.append(words_estimated[best_pos])
            mapped_words_indices.append(best_pos)

    return mapped_words, mapped_words_indices

def get_letter_accuracy(real_word: str, transcribed_word: str) -> list:
    return [1 if i < len(transcribed_word) and r == transcribed_word[i] else 0 for i, r in enumerate(real_word)]

def parse_letter_errors_to_html(word_real: str, is_letter_correct: list) -> str:
    return ''.join([f'<span class="{"correct" if correct else "incorrect"}">{letter}</span>'
                    for letter, correct in zip(word_real, is_letter_correct)])