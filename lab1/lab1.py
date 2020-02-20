import cv2
import os
import numpy as np
import wikipedia
import re

raw_text_pattern = re.compile("[^a-z ]")


def read_images(path, reshape=(100, 100)):
    """
        Reads images from folder, reshapes them an collect into a np.ndarray.
    """
    img_list = os.listdir(path)
    image_tensor = np.empty((len(img_list), *reshape, 3), dtype='i')

    for i, img_name in enumerate(img_list):
        img = cv2.imread(os.path.join(path, img_name))
        image_tensor[i] = cv2.resize(img, (100, 100))
    return image_tensor


def load_pages(path, language='en', num_pages=10, q=False):
    """
        Loads random articles from wikipedia to folder in .txt format.
    """
    wikipedia.set_lang(language)
    if not q:
        print(f'Loading {num_pages} articles to {path}')

    for i in range(1, num_pages+1):
        title = wikipedia.random()
        try:
            with open(os.path.join(path, title + '.txt'), 'w') as f:
                f.write(wikipedia.page(title).content)
            if not q:
                print(f'[{i}/{num_pages}]Loaded {title}.')

        except Exception:
            os.remove(os.path.join(path, title + '.txt'))


def _read_raw_text(filename):
    """
        Reads raw text from single file.
    """
    with open(filename, 'r') as f:
        return raw_text_pattern.sub('', f.read().lower())


def read_raw_text(path):
    """
        Reads raw text from all files in directory.
    """
    res = []
    file_list = os.listdir(path)
    for filename in file_list:
        text = _read_raw_text(os.path.join(path, filename))
        res += text.split()
    return res


def one_hot_encoder(batch, head):
    """
        Performs a one hot encoding for given batch.
    """
    enc = np.zeros((len(batch), len(head)), dtype='i')
    for i, word in enumerate(batch):
        if word in head:
            enc[i][np.where(word == head)] = 1
    return enc
