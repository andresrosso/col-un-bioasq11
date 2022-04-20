import re

ACCENT_MAP = {'À': 'A', 'Á': 'A', 'Â': 'A', 'Ã': 'A', 'Ä': 'A',
             'à': 'a', 'á': 'a', 'â': 'a', 'ã': 'a', 'ä': 'a', 'ª': 'A',
             'È': 'E', 'É': 'E', 'Ê': 'E', 'Ë': 'E',
             'è': 'e', 'é': 'e', 'ê': 'e', 'ë': 'e',
             'Í': 'I', 'Ì': 'I', 'Î': 'I', 'Ï': 'I',
             'í': 'i', 'ì': 'i', 'î': 'i', 'ï': 'i',
             'Ò': 'O', 'Ó': 'O', 'Ô': 'O', 'Õ': 'O', 'Ö': 'O',
             'ò': 'o', 'ó': 'o', 'ô': 'o', 'õ': 'o', 'ö': 'o', 'º': 'O',
             'Ù': 'U', 'Ú': 'U', 'Û': 'U', 'Ü': 'U',
             'ù': 'u', 'ú': 'u', 'û': 'u', 'ü': 'u',
             'Ñ': 'N', 'ñ': 'n',
             'Ç': 'C', 'ç': 'c',
             '§': 'S',  '³': '3', '²': '2', '¹': '1'}
TEXT_NORMALIZATION = str.maketrans(ACCENT_MAP)
NUMBER_REGEX = r'[0-9]'
ALPHANUM_REGEX = r'[^a-zA-Z0-9\s]'
SPACE_REGEX = '\s+'
REPLACEMENT_REGEX = ''
SINGLE_SPACE = ' '

def remove_accents(text):
    return text.translate(TEXT_NORMALIZATION)

def standardize_text(text):
    unaccented_text = remove_accents(text)
    stripped_text = unaccented_text.strip().lower()
    alpha_text = re.sub(NUMBER_REGEX, REPLACEMENT_REGEX, stripped_text)
    return alpha_text

def clean_sentence(sentence):
    unpunctuated_sentence = re.sub(ALPHANUM_REGEX, REPLACEMENT_REGEX, sentence)
    cleaned_sentence = re.sub(SPACE_REGEX, SINGLE_SPACE, unpunctuated_sentence)
    return cleaned_sentence
