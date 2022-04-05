from io import open
import unicodedata
import string
import re
import random
import pickle
import torch

SOS_token = 0
EOS_token = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


# Lowercase, replace contractions, add space before and after punctuation
def normalizeEnglish(s):
    s = s.lower()
    s = decontracted(s)
    s = re.sub(r"([.!?])", r" \1", s) #find puncts .!? and add space beforehand
    return s

# Lowercase, add space before and after punctuation
def normalizeSpanish(s):
    s = s.lower()
    s = decontracted(s)
    s = re.sub(r"([.!?])", r" \1", s) #find puncts .!? and add space beforehand
    s = s.replace('¡', '¡ ') #add space after punctuation
    s = s.replace('¿', '¿ ')
    return s

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


