import re
import sys
import numpy as np
from lab1.lab1 import one_hot_encoder

sys.path.append('..')


def clear_text(text: str):
    text = re.sub(r"\s", ' ', text)
    text = re.sub("[^a-z ]", '', text.lower())
    return text


def bag_of_words(text: str, head: str):
    enc = one_hot_encoder(text.split(), head)
    return np.sum(enc, axis=0)
