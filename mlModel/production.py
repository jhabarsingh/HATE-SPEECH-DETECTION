from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize

from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Dense, Input, Flatten, Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, concatenate
from keras.models import Model, Sequential
from tensorflow.keras import regularizers
from keras.models import load_model


def joiner(file_name):
    paths = os.path.dirname(os.path.abspath(__file__))
    paths = os.path.join(paths, file_name)
    return paths

model = load_model(joiner('pickle.h5'))