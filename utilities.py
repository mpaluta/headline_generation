import numpy as np
import pandas as pd
import csv
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

def load_embeddings(filename):
    """
    Load a DataFrame from pretrained embeddings for sentiment analysis. 
    We are using GloVe.
    """
    labels = []
    rows = []
    with open(filename, encoding='utf-8') as infile:
        for i, line in enumerate(infile):
            items = line.rstrip().split(' ')
            if len(items) == 2:
                # This is a header row giving the shape of the matrix
                continue
            labels.append(items[0])
            values = np.array([float(x) for x in items[1:]], 'f')
            rows.append(values)
    return pd.DataFrame(np.vstack(rows), index=labels, dtype='f')

def load_lexicon(filename):
    """
    Load a file from Bing Liu's sentiment lexicon
    (https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html), containing
    English words in Latin-1 encoding.
    
    One file contains a list of positive words, and the other contains
    a list of negative words. The files contain comment lines starting
    with ';' and blank lines, which should be skipped.
    """
    lexicon = []
    with open(filename, encoding='latin-1') as infile:
        for line in infile:
            line = line.rstrip()
            if line and not line.startswith(';'):
                lexicon.append(line)
    return lexicon

class data_setup():

    @staticmethod
    def get_paths(set="train"):
        with open(set+"_IDs.csv") as file_path:
            reader = csv.reader(file_path)
            paths = list(reader)
        if [] in paths:
            paths = paths[::2]
        return paths

    